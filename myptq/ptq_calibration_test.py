"""Calibration/weight composition regression fixtures; actual CUDA is checked remotely."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import numpy.testing
import sklearn.metrics
import torch
from torch import nn
from torch.nn import functional as F
import ptq
import ptq_quant as weights
import ptq_calibration as activation
import ptq_sweep as sweep

REPO=Path(os.environ.get('MYPTQ_TEST_REPO',str(Path(__file__).resolve().parents[1])))

def reference_scan(u,delta,A,B,C,D,z=None,delta_bias=None,delta_softplus=False,return_last_state=False):
    # Independent sequential recurrence; production always calls the CUDA scan.
    dt=delta+delta_bias[None,:,None]
    if delta_softplus:dt=F.softplus(dt)
    state=torch.zeros(u.shape[0],u.shape[1],A.shape[1],device=u.device,dtype=u.dtype);values=[]
    for t in range(u.shape[-1]):
        state=state*torch.exp(dt[:,:,t,None]*A[None])+dt[:,:,t,None]*B[:,None,:,t]*u[:,:,t,None]
        values.append((state*C[:,None,:,t]).sum(-1)+D[None]*u[:,:,t])
    output=torch.stack(values,dim=-1)
    if z is not None:output=output*F.silu(z)
    return (output,state) if return_last_state else output

class FixtureMamba(nn.Module):
    def __init__(self,d_model=4,expand=2):
        super().__init__();self.d_model=d_model;self.d_inner=expand*d_model;self.d_state=2
        self.dt_rank=(d_model+15)//16;self.activation='silu';self.use_fast_path=True
        self.in_proj=nn.Linear(d_model,2*self.d_inner,bias=False)
        self.conv1d=nn.Conv1d(self.d_inner,self.d_inner,4,padding=3,groups=self.d_inner)
        self.x_proj=nn.Linear(self.d_inner,self.dt_rank+2*self.d_state,bias=False)
        self.dt_proj=nn.Linear(self.dt_rank,self.d_inner,bias=True)
        self.dt_proj.bias.data.fill_(-2)
        self.A_log=nn.Parameter(torch.arange(1,3,dtype=torch.float32).log().repeat(self.d_inner,1))
        self.D=nn.Parameter(torch.ones(self.d_inner))
        self.out_proj=nn.Linear(self.d_inner,d_model,bias=False)

    def forward(self,x,inference_params=None):
        n,length,_=x.shape
        # Stock-style unfused reference, using different layout/linear calls.
        packed=(self.in_proj.weight@x.reshape(-1,self.d_model).T).reshape(2*self.d_inner,n,length).permute(1,0,2)
        u,z=packed.chunk(2,dim=1)
        u=F.silu(self.conv1d(u)[...,:length])
        projected=self.x_proj(u.transpose(1,2).reshape(-1,self.d_inner))
        dt,B,C=projected.split([self.dt_rank,2,2],dim=-1)
        dt=(self.dt_proj.weight@dt.T).reshape(self.d_inner,n,length).permute(1,0,2)
        B=B.reshape(n,length,2).transpose(1,2);C=C.reshape(n,length,2).transpose(1,2)
        y=reference_scan(u,dt,-self.A_log.exp(),B,C,self.D,z=z,delta_bias=self.dt_proj.bias,delta_softplus=True)
        return self.out_proj(y.transpose(1,2))

class FixtureBidirectional(nn.Module):
    def __init__(self):
        super().__init__();self.bidirectional=True;self.bidirectional_strategy='add'
        self.mamba_fwd=FixtureMamba();self.mamba_rev=FixtureMamba()
    def forward(self,x):return self.mamba_fwd(x)+self.mamba_rev(x.flip(1)).flip(1)

class FixturePatch(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(1,4,(2,4),stride=(2,4))
    def forward(self,x):return self.proj(x[:,None]).flatten(1,2).transpose(1,2)

class FixtureHead(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(4,4);self.activation1=nn.GELU()
        self.mamba_1=FixtureMamba();self.fc3=nn.Linear(4,2)
    def forward(self,x):return self.fc3(self.mamba_1(self.activation1(self.fc1(x))).transpose(1,2).contiguous().mean(-1))

class FixtureFEMBA(nn.Module):
    def __init__(self):
        super().__init__();self.classification_type='bc';self.num_classes=2
        self.patch_embed=FixturePatch();self.pos_embed=nn.Parameter(torch.randn(1,8,4)*.01)
        self.mamba_blocks=nn.ModuleList([FixtureBidirectional(),FixtureBidirectional()])
        self.norm_layers=nn.ModuleList([nn.LayerNorm(4),nn.LayerNorm(4)]);self.classifier=FixtureHead()
    def forward(self,x,mask):
        original=x;masked=x.clone();masked[mask]=0
        x=self.patch_embed(masked)+self.pos_embed
        for block,norm in zip(self.mamba_blocks,self.norm_layers):x=norm(x+block(x))
        return self.classifier(x),original

class CalibrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
    def model(self):
        model=FixtureFEMBA().eval();model.classifier.classification_type='bc';return model
    def context(self):
        stack=contextlib.ExitStack()
        root=types.ModuleType('mamba_ssm');root.__path__=[];root.Mamba=FixtureMamba
        ops=types.ModuleType('mamba_ssm.ops');ops.__path__=[]
        interface=types.ModuleType('mamba_ssm.ops.selective_scan_interface');interface.selective_scan_fn=reference_scan
        replacements={'mamba_ssm':root,'mamba_ssm.ops':ops,'mamba_ssm.ops.selective_scan_interface':interface}
        old={name:sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)
        def restore_modules():
            for name,prior in old.items():
                if prior is None:sys.modules.pop(name,None)
                else:sys.modules[name]=prior
        stack.callback(restore_modules)
        def build(config,state,repo):
            model=self.model()
            if state is not None:model.load_state_dict(state,strict=True)
            return model
        stack.enter_context(patch.object(ptq,'build_model',side_effect=build))
        def install(model,config):return activation.install_activations(model,config,mamba_type=FixtureMamba,scan_fn=reference_scan)
        stack.enter_context(patch.object(ptq,'install_activations',side_effect=install))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack
    def data(self,root):
        for i,split in enumerate(('train','val','test')):
            n=40 if split=='train' else 12
            with h5py.File(root/(split+'.h5'),'w') as h:
                group=h.create_group('part');group.create_dataset('X',data=np.random.default_rng(i).normal(0,5,(n,2,32)).astype('float32'));group.create_dataset('y',data=np.arange(n)%2)
        ckpt=root/'float.ckpt';model=self.model()
        torch.save({'model_config':{'num_classes':2,'classification_type':'bc'},'normalization':{'enabled':True,'lower':-20,'upper':20},'state_dict':model.state_dict()},ckpt)
        return ckpt
    def command(self,root,ckpt,out):
        return ['--repo',str(REPO),'--checkpoint',str(ckpt),'--data-dir',str(root),'--output-dir',str(out),'--device','cpu','--batch-size','4']
    def test_functional_options_and_profile_defaults(self):
        args=ptq.parser().parse_args([]);self.assertFalse(hasattr(args,'backend'))
        self.assertEqual(ptq.configuration(args).activation_bits,32)
        for method,windows,batch,seed,state in [('observer',512,32,42,'float'),('sequential',1024,128,0,'quantized')]:
            cfg=ptq.configuration(ptq.parser().parse_args(['--activation-bits','4','--calibration-method',method]))
            self.assertEqual((cfg.calibration_windows,cfg.calibration_batch_size,cfg.calibration_seed,cfg.calibration_weight_state),(windows,batch,seed,state))
    def test_group_and_tensor_mse_never_worse_than_absmax(self):
        rows=torch.randn(3,385,dtype=torch.float64);rows[:,0]=10
        for bits in (4,8):
            for group in (32,64,128,None):
                a,_=weights.quantize_rows(rows,bits,group);b,_=weights.quantize_rows(rows,bits,group,initialization='mse')
                self.assertLessEqual(float((b-rows).square().sum()),float((a-rows).square().sum())+1e-10)
            a,_=weights.quantize_rows(rows,bits,None,'tensor');b,_=weights.quantize_rows(rows,bits,None,'tensor',initialization='mse')
            self.assertLessEqual(float((b-rows).square().sum()),float((a-rows).square().sum())+1e-10)
        with self.assertRaises(ValueError):weights.PTQConfig(weight_init='mse',clip_percentile=99).validate()
    def test_activation_layouts_preserve_disabled_float_computation(self):
        with self.context():
            for method,count in [('observer',42),('sequential',32)]:
                model=self.model();cfg=weights.PTQConfig(activation_bits=4,calibration_method=method)
                x=torch.randn(2,2,32);mask=torch.zeros_like(x,dtype=torch.bool);before=model(x,mask)[0]
                controller=activation.install_activations(model,cfg,mamba_type=FixtureMamba,scan_fn=reference_scan)
                torch.testing.assert_close(model(x,mask)[0],before,rtol=3e-4,atol=1e-4)
                self.assertEqual(len(controller.names),count)
    def test_indices_are_fixed_without_replacement(self):
        for method,seed in [('observer',42),('sequential',0)]:
            cfg=weights.PTQConfig(activation_bits=4,calibration_method=method,calibration_seed=seed,calibration_windows=8)
            actual=activation.choose_indices(40,cfg)
            expected=torch.randperm(40,generator=torch.Generator().manual_seed(42))[:8].tolist() if method=='observer' else sorted(np.random.default_rng(0).choice(40,8,replace=False).tolist())
            self.assertEqual(actual,expected)
    def test_rotation_mse_and_sequential_calibration_compose(self):
        with tempfile.TemporaryDirectory(prefix='myptq-compose-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root);out=root/'result'
            argv=self.command(root,ckpt,out)+['--scope','all-weights','--group-size','32','--method','block','--rotation-size','32',
                 '--rotation-transform','hadamard','--weight-init','mse','--activation-bits','4','--calibration-method','sequential',
                 '--activation-percentile','95','--calibration-windows','8']
            ptq.main(argv)
            meta=json.loads((out/'myptq_metadata.json').read_text());self.assertTrue(meta['rotation_equivalence']['checked'])
            self.assertEqual(meta['activation_quantizers'],32);self.assertEqual(meta['ptq']['quantized_weight_tensors'],28)
            self.assertEqual(meta['calibration_report']['stages'],24);self.assertTrue(meta['strict_load'])
    def test_observer_uses_declared_weight_state(self):
        with tempfile.TemporaryDirectory(prefix='myptq-weight-state-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root);original=torch.load(ckpt,weights_only=False)['state_dict']['patch_embed.proj.weight']
            real=ptq.calibrate
            for state in ('float','quantized'):
                def inspect(model,controller,config,batches):
                    same=torch.equal(model.patch_embed.proj.weight.detach().cpu(),original)
                    self.assertEqual(same,state=='float')
                    return real(model,controller,config,batches)
                with patch.object(ptq,'calibrate',side_effect=inspect):
                    ptq.main(self.command(root,ckpt,root/state)+['--scope','all-weights','--activation-bits','8',
                         '--calibration-method','observer','--calibration-windows','8','--calibration-weight-state',state])
                meta=json.loads((root/state/'myptq_metadata.json').read_text())
                self.assertEqual(meta['activation_quantizers'],42)
    def test_frozen_config_is_automatic_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory(prefix='myptq-frozen-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root);out=root/'val-result'
            ptq.main(self.command(root,ckpt,out)+['--scope','all-weights','--group-size','none','--weight-init','mse',
                         '--activation-bits','4','--calibration-method','sequential','--activation-percentile','95','--calibration-windows','8'])
            saved=out/'myptq.ckpt';test=root/'test-result'
            command=self.command(root,saved,test)+['--load-calibrated','--split','test']
            ptq.main(command)
            meta=json.loads((test/'myptq_metadata.json').read_text());self.assertEqual(meta['config']['weight_init'],'mse')
            self.assertEqual(meta['config']['activation_bits'],4);self.assertIsNone(meta['calibration_data'])
            with self.assertRaisesRegex(ValueError,'conflicts'):ptq.main(self.command(root,saved,root/'bad')+['--load-calibrated','--bits','8'])
    def test_weight_only_activation_only_and_float(self):
        with tempfile.TemporaryDirectory(prefix='myptq-components-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root)
            for bits,act in ((4,32),(0,8),(0,32)):
                out=root/f'w{bits}a{act}'
                ptq.main(self.command(root,ckpt,out)+['--bits',str(bits),'--activation-bits',str(act),'--calibration-windows','8'])
                self.assertEqual((out/'myptq.ckpt').exists(),bool(bits or act!=32))
                meta=json.loads((out/'myptq_metadata.json').read_text());self.assertEqual(meta['activation_quantized'],act!=32)
    def test_mixed_functional_sweep_deduplicates_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix='myptq-options-grid-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root);out=root/'grid'
            argv=['--repo',str(REPO),'--checkpoint',str(ckpt),'--data-dir',str(root),'--output-dir',str(out),
                  '--group-sizes','none','--methods','symmetric','--weight-inits','max','mse','--activation-bit-widths','32','4',
                  '--calibration-methods','observer','sequential','--activation-percentiles','95','--calibration-windows','8',
                  '--device','cpu','--batch-size','4','--run']
            def run(command,**kwargs):
                with contextlib.redirect_stdout(kwargs['stdout']):ptq.main(command[2:])
            with patch.object(sweep.subprocess,'run',side_effect=run) as runner:
                rows=sweep.main(argv);self.assertEqual(len(rows),6);self.assertEqual(runner.call_count,6)
                sweep.main(argv);self.assertEqual(runner.call_count,6)
            best=json.loads((out/'myptq_best_validation.json').read_text());self.assertEqual(Path(best['checkpoint_file']).name,'myptq.ckpt')
    def test_partial_evaluation_is_marked(self):
        with tempfile.TemporaryDirectory(prefix='myptq-partial-') as folder,self.context():
            root=Path(folder);ckpt=self.data(root);out=root/'result'
            ptq.main(self.command(root,ckpt,out)+['--max-batches','1'])
            self.assertTrue(json.loads((out/'myptq_complete.json').read_text())['partial_evaluation'])


if __name__=='__main__':unittest.main(verbosity=2)
