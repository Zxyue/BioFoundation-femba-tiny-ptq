#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

"""Regression tests for myptq math, CLI grid and an HDF5 evaluation fixture."""
import argparse
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ptq
import ptq_quant as quant
import ptq_sweep as sweep


class ToyFEMBA(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Linear(16,25)
        self.mamba_blocks=nn.ModuleList([nn.ModuleDict({'in_proj':nn.Linear(25,25),'out_proj':nn.Linear(25,25),'conv1d':nn.Conv1d(25,25,3,groups=25)})])
        self.classifier=nn.Sequential(nn.Linear(25,2))

    def forward(self,x,mask):
        y=self.patch_embed(x)
        block=self.mamba_blocks[0]
        y=F.linear(y,block['in_proj'].weight,block['in_proj'].bias)
        y=F.gelu(y)
        y=(block['out_proj'].weight @ y.reshape(-1,25).T).T.reshape(y.shape)+block['out_proj'].bias
        return self.classifier(y.mean(1)),x


def hadamard(n):
    h=torch.ones(1,1,dtype=torch.float64)
    while len(h)<n:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(n)


def dense_rotation(rotation):
    if rotation.method=='symmetric':return torch.eye(rotation.width,dtype=torch.float64)
    if rotation.transform=='qr':blocks=list(rotation.matrix.unbind(0))
    else:
        h=hadamard(rotation.block_size)
        blocks=[torch.diag(sign)@h for sign in rotation.signs]
    return torch.block_diag(*blocks)[:rotation.width,:rotation.active_width]


class MyPTQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fwht_dense_and_inverse(self):
        for width in (1,2,32,64,128,512):
            x=torch.randn(3,width,dtype=torch.float64)
            torch.testing.assert_close(quant.fwht(x),x@hadamard(width),rtol=2e-13,atol=2e-13)
            torch.testing.assert_close(quant.fwht(quant.fwht(x)),x,rtol=2e-13,atol=2e-13)

    def test_every_rotation_matches_independent_dense_matrix(self):
        for width in (1,25,64,65,385):
            for method in ('symmetric','block','full'):
                for transform in ('qr','hadamard'):
                    with self.subTest(width=width,method=method,transform=transform):
                        rotation=quant.Rotation(width,method,transform,64,42)
                        r=dense_rotation(rotation)
                        x=torch.randn(3,width,dtype=torch.float64)
                        torch.testing.assert_close(rotation.forward(x),x@r,rtol=3e-13,atol=3e-13)
                        torch.testing.assert_close(rotation.inverse(x@r),x,rtol=3e-13,atol=3e-13)

    def test_large_non_multiple_block_rotations(self):
        for size in (32,64,128):
            for transform in ('qr','hadamard'):
                rotation=quant.Rotation(1540,'block',transform,size,42)
                x=torch.randn(3,1540,dtype=torch.float64)
                torch.testing.assert_close(rotation.inverse(rotation.forward(x)),x,rtol=3e-13,atol=3e-13)

    def test_hadamard_keeps_padded_coordinates_until_inverse(self):
        for method,expected in [('block',64),('full',32)]:
            r=quant.Rotation(25,method,'hadamard',64,42)
            x=torch.randn(2,25,dtype=torch.float64)
            y=r.forward(x)
            self.assertEqual(y.shape[-1],expected)
            self.assertGreater(torch.count_nonzero(y[:,25:]).item(),0)
            torch.testing.assert_close(r.inverse(y),x,rtol=1e-13,atol=1e-13)

    def test_rotation_global_rng_isolation(self):
        state=torch.get_rng_state().clone()
        quant.Rotation(385,'full','qr',64,42)
        quant.Rotation(385,'full','hadamard',64,42)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_layer_seed_is_stable_and_order_independent(self):
        a=quant.layer_seed(42,'mamba_blocks.0.in_proj')
        self.assertEqual(a,quant.layer_seed(42,'mamba_blocks.0.in_proj'))
        self.assertNotEqual(a,quant.layer_seed(42,'mamba_blocks.1.in_proj'))

    def test_quantization_granularity_scale_counts(self):
        w=torch.randn(3,385,dtype=torch.float64)
        for size in (32,64,128,None):
            q,stats=quant.quantize_rows(w,4,size)
            self.assertEqual(q.shape,w.shape)
            self.assertEqual(stats['scales'],3*(math.ceil(385/size) if size else 1))
        _,stats=quant.quantize_rows(w,4,None,'tensor')
        self.assertEqual(stats['scales'],1)

    def test_group_quantizer_independent_reference(self):
        w=torch.randn(2,385,dtype=torch.float64)
        for bits in (4,8):
            for size in (32,64,128,None):
                expected=w.clone();step=size or 385;qmax=2**(bits-1)-1
                for row in range(2):
                    for start in range(0,385,step):
                        part=w[row,start:start+step];scale=max(float(part.abs().max())/qmax,1e-12)
                        expected[row,start:start+step]=(part/scale).round().clamp(-qmax,qmax)*scale
                actual,_=quant.quantize_rows(w,bits,size)
                torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)

    def test_partial_group_percentile_ignores_padding(self):
        w=torch.zeros(2,65,dtype=torch.float64);w[:,-1]=10
        q,_=quant.quantize_rows(w,4,64,percentile=50)
        torch.testing.assert_close(q[:,-1],w[:,-1],rtol=0,atol=0)

    def test_effective_weights_match_explicit_rotation_for_every_group(self):
        w=torch.randn(5,65,dtype=torch.float64);x=torch.randn(2,3,65,dtype=torch.float64)
        for method in ('symmetric','block','full'):
            for transform in ('qr','hadamard'):
                for size in (32,64,128,None):
                    with self.subTest(method=method,transform=transform,group=size):
                        cfg=quant.PTQConfig(group_size=size,method=method,rotation_transform=transform)
                        rotation=quant.Rotation(65,method,transform,64,42);r=dense_rotation(rotation)
                        q,_=quant.quantize_rows(w@r,4,size)
                        expected=F.linear(x@r,q)
                        effective,_=quant.effective_weight(w,cfg,42)
                        torch.testing.assert_close(F.linear(x,effective),expected,rtol=4e-13,atol=4e-13)

    def test_float_diagnostics_are_equivalent(self):
        w=torch.randn(5,385)
        for method in ('block','full'):
            for transform in ('qr','hadamard'):
                effective,_=quant.effective_weight(w,quant.PTQConfig(bits=0,method=method,rotation_transform=transform),42)
                torch.testing.assert_close(effective,w,rtol=0,atol=1e-12)

    def test_zero_and_nonfinite_weights(self):
        q,_=quant.effective_weight(torch.zeros(3,65),quant.PTQConfig(method='full',rotation_transform='hadamard'),42)
        self.assertEqual(torch.count_nonzero(q).item(),0)
        with self.assertRaises(ValueError):quant.quantize_rows(torch.tensor([[float('nan')]]))

    def test_tensor_requires_no_grouping(self):
        with self.assertRaises(ValueError):quant.PTQConfig(granularity='tensor',group_size=64).validate()

    def test_invalid_configuration_rejected(self):
        for settings in ({'group_size':16},{'clip_percentile':0},{'clip_percentile':float('nan')},
                         {'rotation_size':256},{'rotation_seed':-1},{'bits':3},{'scope':'bad'}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):quant.PTQConfig(**settings).validate()

    def test_preserve_module_parameter_identity_and_scope(self):
        model=ToyFEMBA().eval();before=copy.deepcopy(model.state_dict())
        identity={n:(id(m),id(m.weight)) for n,m in model.named_modules() if isinstance(m,(nn.Linear,nn.Conv1d))}
        report=quant.apply_ptq(model,quant.PTQConfig(method='block',rotation_transform='hadamard'))
        self.assertEqual(report['quantized_weight_tensors'],2)
        self.assertEqual(identity,{n:(id(m),id(m.weight)) for n,m in model.named_modules() if isinstance(m,(nn.Linear,nn.Conv1d))})
        changed={r['name']+'.weight' for r in report['layers']}
        for name,value in model.state_dict().items():
            if name not in changed:self.assertTrue(torch.equal(value,before[name]),name)

    def test_all_weights_scope_quantizes_conv_without_rotating_it(self):
        model=ToyFEMBA();report=quant.apply_ptq(model,quant.PTQConfig(scope='all-weights',method='full',rotation_transform='hadamard'))
        conv=next(r for r in report['layers'] if r['module_type']=='Conv1d')
        self.assertEqual(conv['rotation']['method'],'symmetric')
        self.assertEqual(report['quantized_weight_tensors'],5)

    def test_classifier_override(self):
        model=ToyFEMBA();report=quant.apply_ptq(model,quant.PTQConfig(scope='all-linear',classifier_bits=8))
        self.assertEqual(next(r['bits'] for r in report['layers'] if r['name'].startswith('classifier.')),8)
        report=quant.apply_ptq(ToyFEMBA(),quant.PTQConfig(scope='all-linear',classifier_bits=0))
        self.assertFalse(any(r['name'].startswith('classifier.') for r in report['layers']))

    def test_include_and_exclude_patterns(self):
        cfg=quant.PTQConfig(scope='all-linear',method='full',rotation_transform='hadamard',include=('classifier.0',))
        report=quant.apply_ptq(ToyFEMBA(),cfg)
        self.assertEqual([r['name'] for r in report['layers']],['classifier.0'])
        cfg=quant.PTQConfig(exclude=('*.out_proj',))
        report=quant.apply_ptq(ToyFEMBA(),cfg)
        self.assertEqual([r['name'] for r in report['layers']],['mamba_blocks.0.in_proj'])

    def test_duplicate_and_parametrized_weights_rejected(self):
        model=ToyFEMBA();quant.apply_ptq(model,quant.PTQConfig())
        with self.assertRaises(RuntimeError):quant.apply_ptq(model,quant.PTQConfig())
        model=ToyFEMBA();nn.utils.parametrize.register_parametrization(model.mamba_blocks[0]['in_proj'],'weight',nn.Identity())
        with self.assertRaises(ValueError):quant.apply_ptq(model,quant.PTQConfig())

    def test_float_diagnostic_model_with_direct_weight_reads(self):
        model=ToyFEMBA().eval();other=copy.deepcopy(model);x=torch.randn(2,2,16)
        quant.apply_ptq(other,quant.PTQConfig(bits=0,method='full',rotation_transform='hadamard'))
        torch.testing.assert_close(other(x,None)[0],model(x,None)[0],rtol=0,atol=1e-7)

    def test_state_dict_reload_compatibility(self):
        model=ToyFEMBA().eval();quant.apply_ptq(model,quant.PTQConfig(method='block'))
        other=ToyFEMBA().eval();other.load_state_dict(model.state_dict(),strict=True);x=torch.randn(2,2,16)
        torch.testing.assert_close(other(x,None)[0],model(x,None)[0],rtol=0,atol=0)

    def test_cli_parameter_semantics(self):
        p=ptq.parser()
        cfg=ptq.configuration(p.parse_args(['--group-size','none','--method','full','--rotation-transform','hadamard']))
        self.assertIsNone(cfg.group_size)
        with self.assertRaises(ValueError):ptq.configuration(p.parse_args(['--method','full','--rotation-size','64']))

    def test_default_sweep_has_36_nonduplicated_cases(self):
        args=argparse.Namespace(group_sizes=[32,64,128,None],granularity='channel',methods=['symmetric','block','full'],
             rotation_transforms=['qr','hadamard'],rotation_sizes=[32,64,128],rotation_seeds=[2025],clip_percentiles=[100.],
             bits=4,scope='encoder-linear',classifier_bits=None,output_dir=Path('/out'),repo=Path('/repo'),
             checkpoint=Path('/float.ckpt'),data_dir=Path('/data'),data_file=None,device='cuda',batch_size=64,max_batches=0,experiment='FEMBA_tiny_finetune')
        jobs=sweep.grid(args)
        self.assertEqual(len(jobs),36)
        self.assertEqual(len({j['name'] for j in jobs}),36)
        self.assertEqual(sum(j['config']['method']=='symmetric' for j in jobs),4)
        for j in jobs:
            child=ptq.parser().parse_args(j['argv'][2:])
            self.assertEqual(ptq.configuration(child).to_dict(),j['config'])
            self.assertEqual(child.split,'val')

    def test_metrics_use_margin_and_keep_legacy_score_separately(self):
        metrics=ptq.compute_metrics(np.array([[10.,9.],[0.,2.]]),np.array([0,1]))
        self.assertEqual(metrics['auroc'],1.)
        self.assertEqual(metrics['auroc_legacy_positive_logit'],0.)
        with self.assertRaises(ValueError):ptq.compute_metrics([[1,0],[2,0]],[0,0])

    def test_float_checkpoint_metadata_and_qat_rejection(self):
        cfg={'model':{'_target_':'models.FEMBA.FEMBA','embed_dim':35,'num_blocks':2,'num_classes':2},
             'input_normalization':{'normalize':True,'quartile_normalization_lower_val':-20,'quartile_normalization_upper_val':20}}
        model,norm,state=ptq.checkpoint_spec({'hyper_parameters':cfg,'state_dict':{'model.a':torch.ones(2)}},Path('/repo'),'unused')
        self.assertEqual(model['num_blocks'],2);self.assertEqual(norm['lower'],-20);self.assertEqual(set(state),{'a'})
        with self.assertRaises(ValueError):ptq.checkpoint_spec({'stage':'ptq'},Path('/repo'),'unused')
        with self.assertRaises(ValueError):ptq.checkpoint_spec({'state_dict':{'model.x.parametrizations.weight.original':torch.ones(1)}},Path('/repo'),'unused')

    def test_hdf5_evaluation_cli_integration(self):
        # Real Torch/HDF5/normalization/metrics/output; only the CUDA-dependent FEMBA constructor is replaced.
        repo=Path(os.environ.get('MYPTQ_TEST_REPO',str(Path(__file__).resolve().parents[1])))
        with tempfile.TemporaryDirectory(prefix='myptq-') as directory:
            temp=Path(directory);data=temp/'val.h5';checkpoint=temp/'float.ckpt';out=temp/'result'
            rng=np.random.default_rng(42)
            with h5py.File(data,'w') as h:
                group=h.create_group('part');group.create_dataset('X',data=rng.normal(size=(16,2,16)).astype('float32'));group.create_dataset('y',data=np.arange(16)%2)
            model=ToyFEMBA().eval()
            payload={'model_config':{'num_classes':2,'num_blocks':2,'num_channels':2,'seq_length':16},
                     'normalization':{'enabled':True,'lower':-20,'upper':20},'state_dict':model.state_dict()}
            torch.save(payload,checkpoint)
            def build(config,state,path):
                result=ToyFEMBA().eval();result.load_state_dict(state,strict=True);return result
            argv=['--repo',str(repo),'--checkpoint',str(checkpoint),'--data-file',str(data),'--output-dir',str(out),
                  '--method','block','--rotation-transform','hadamard','--rotation-size','32','--device','cpu','--batch-size','4']
            with patch.object(ptq,'build_model',side_effect=build),contextlib.redirect_stdout(io.StringIO()):ptq.main(argv)
            metrics=json.loads((out/'myptq_metrics.json').read_text())
            metadata=json.loads((out/'myptq_metadata.json').read_text())
            self.assertEqual(metrics['samples'],16);self.assertTrue(metadata['rotation_equivalence']['checked'])
            self.assertEqual(metadata['ptq']['quantized_weight_tensors'],2)
            with np.load(out/'myptq_predictions.npz') as predictions:self.assertEqual(predictions['logits'].shape,(16,2))
            self.assertTrue((out/'myptq_complete.json').is_file())
            with self.assertRaises(FileExistsError):ptq.main(argv)


    def test_sweep_dispatch_and_completed_case_resume(self):
        repo=Path(os.environ.get('MYPTQ_TEST_REPO',str(Path(__file__).resolve().parents[1])))
        with tempfile.TemporaryDirectory(prefix='myptq-grid-') as directory:
            temp=Path(directory);data=temp/'val.h5';checkpoint=temp/'float.ckpt';out=temp/'sweep'
            with h5py.File(data,'w') as h:
                group=h.create_group('part');group.create_dataset('X',data=np.random.default_rng(0).normal(size=(12,2,16)).astype('float32'));group.create_dataset('y',data=np.arange(12)%2)
            torch.save({'model_config':{'num_classes':2},'normalization':{'enabled':False},'state_dict':ToyFEMBA().state_dict()},checkpoint)
            def build(config,state,path):
                result=ToyFEMBA().eval();result.load_state_dict(state,strict=True);return result
            def child(argv,**kwargs):
                with contextlib.redirect_stdout(kwargs['stdout']):ptq.main(argv[2:])
            argv=['--repo',str(repo),'--checkpoint',str(checkpoint),'--data-file',str(data),'--output-dir',str(out),
                  '--group-sizes','64','none','--methods','symmetric','block','--rotation-transforms','hadamard',
                  '--rotation-sizes','32','--device','cpu','--batch-size','4','--run']
            with patch.object(ptq,'build_model',side_effect=build),patch.object(sweep.subprocess,'run',side_effect=child) as run,contextlib.redirect_stdout(io.StringIO()):
                result=sweep.main(argv)
                self.assertEqual(len(result),4);self.assertEqual(run.call_count,4)
                sweep.main(argv)
                self.assertEqual(run.call_count,4)
            self.assertTrue((out/'myptq_best_validation.json').exists())
            self.assertEqual(len(list(out.glob('*/myptq_complete.json'))),4)


if __name__=='__main__':unittest.main(verbosity=2)
