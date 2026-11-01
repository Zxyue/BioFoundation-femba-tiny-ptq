"""Observer-only activation calibration on original FEMBA CUDA operations.

Derived from the previous original-FEMBA adaptation. Mamba equations follow
state-spaces/mamba v2.2.6.post3, Copyright (c) 2023 Tri Dao, Albert Gu;
see ptq_mamba_NOTICE.txt. Weight quantization is managed independently.
"""
import types
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from ptq_runtime import load_mamba
CONTROLLER='_myptq_activations'

class Quantizers(nn.Module):
    def __init__(self,model,config,points,scan_fn):
        super().__init__()
        from brevitas.nn import QuantIdentity
        from brevitas.quant import Int8ActPerTensorFloat
        self.config=config;self.enabled=False;self.scan_fn=scan_fn
        object.__setattr__(self,'_model_ref',weakref.ref(model))
        self.names={name:f'a{i:03d}' for i,name in enumerate(points)}
        self.activations=nn.ModuleDict()
        for name,key in self.names.items():
            quant=QuantIdentity(act_quant=Int8ActPerTensorFloat,bit_width=config.activation_bits,return_quant_tensor=False)
            scaling=quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
            scaling.stats.stats_impl.q=config.activation_percentile
            self.activations[key]=quant
    def quant(self,name,x):return self.activations[self.names[name]](x) if self.enabled else x
    def set_enabled(self,enabled):
        from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
        for proxy in self.modules():
            if isinstance(proxy,ActQuantProxyFromInjectorBase):proxy.disable_quant=not enabled
        self.enabled=enabled;self._model_ref().eval()
    def observers(self):
        return {name:self.activations[key].act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl for name,key in self.names.items()}
    def frozen_scales(self):
        if not self.enabled:raise RuntimeError('Activation quantization is disabled.')
        result={}
        for name,key in self.names.items():
            proxy=self.activations[key].act_quant;scaling=self.observers()[name]
            scale=proxy.scale()
            if scaling.counter<=scaling.collect_stats_steps or proxy.bit_width().item()!=self.config.activation_bits:
                raise RuntimeError('Activation observer is not frozen or has the wrong bit width.')
            if scale.numel()!=1 or not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):raise RuntimeError('Invalid activation scale.')
            result[name]=scale.detach().cpu().clone()
        return result

def _mamba_forward(module,x,controller,name,inference_params=None):
    """Expose quantization boundaries while retaining the original CUDA SSM recurrence.

    dt_proj.bias is supplied ONLY to selective_scan (before its softplus), never
    to the delta projection. Weight coverage is controlled by the weight stage.
    """
    if inference_params is not None:raise ValueError('PTQ evaluation supports complete windows, not cached decoding.')
    b,length,_=x.shape
    packed=module.in_proj(x.reshape(b*length,module.d_model)).reshape(b,length,2*module.d_inner)
    packed=controller.quant(name+'.in_proj',packed)
    u,z=packed.transpose(1,2).chunk(2,dim=1)
    convolved=F.conv1d(u,module.conv1d.weight,module.conv1d.bias,
                      stride=module.conv1d.stride,padding=module.conv1d.padding,
                      dilation=module.conv1d.dilation,groups=module.conv1d.groups)[...,:length]
    convolved=controller.quant(name+'.conv',convolved)
    u=controller.quant(name+'.silu',F.silu(convolved))
    projected=module.x_proj(u.transpose(1,2).reshape(b*length,module.d_inner))
    delta,b_state,c_state=projected.split([module.dt_rank,module.d_state,module.d_state],dim=-1)
    # Weights are cached in the original basis by the unified weight stage.
    delta_weight=module.dt_proj.weight
    delta=F.linear(delta,delta_weight,None).reshape(b,length,module.d_inner).transpose(1,2).contiguous()
    b_state=b_state.reshape(b,length,module.d_state).transpose(1,2).contiguous()
    c_state=c_state.reshape(b,length,module.d_state).transpose(1,2).contiguous()
    y=controller.scan_fn(u.contiguous(),delta,-module.A_log.float().exp(),b_state,c_state,
                         module.D.float(),z=z.contiguous(),delta_bias=module.dt_proj.bias.float(),
                         delta_softplus=True,return_last_state=False)
    y=controller.quant(name+'.ssm',y)
    out=module.out_proj(y.transpose(1,2).reshape(b*length,module.d_inner)).reshape(b,length,module.d_model)
    return controller.quant(name+'.out_proj',out)

def install_quantizers(model,config,*,mamba_type=None,scan_fn=None):
    """Call AFTER strict float checkpoint loading. Original Parameters are reused.

    Explicit runtime injections are for deterministic unit fixtures only. The
    normal path imports the pinned Mamba class and its CUDA selective_scan_fn.
    """
    config.validate()
    import brevitas
    if brevitas.__version__!='0.12.1':raise RuntimeError('Expected brevitas==0.12.1.')
    if hasattr(model,CONTROLLER):raise ValueError('Activation quantizers already installed.')
    if mamba_type is None or scan_fn is None:
        runtime_type,runtime_scan=load_mamba()
        mamba_type=mamba_type or runtime_type;scan_fn=scan_fn or runtime_scan
    if model.classification_type!='bc' or model.num_classes!=2:raise ValueError('This adapter currently supports binary FEMBA classification.')
    head=model.classifier
    if not all(hasattr(head,k) for k in ('fc1','activation1','mamba_1','fc3')):raise ValueError('Expected the original FEMBA MambaClassifier head.')
    mambas=[(n,m) for n,m in model.named_modules() if isinstance(m,mamba_type)]
    blocks=list(model.mamba_blocks)
    if len(mambas)!=2*len(blocks)+1:raise ValueError('Expected bidirectional encoder Mambas and the original classifier Mamba.')
    for block in blocks:
        if not block.bidirectional or block.bidirectional_strategy!='add':raise ValueError('Expected bidirectional add encoder blocks.')
    points=['input','patch','position','embed_add']
    for name,_ in mambas:points += [name+'.'+suffix for suffix in ('in_proj','conv','silu','ssm','out_proj')]
    for i in range(len(blocks)):
        points += [f'mamba_blocks.{i}.'+s for s in ('reverse_input','reverse_output','equalize','sum')]
        points += [f'norm_layers.{i}']
    points += ['classifier.fc1','classifier.gelu','classifier.pool']
    controller=Quantizers(model,config,points,scan_fn)
    model.add_module(CONTROLLER,controller)
    ref=weakref.ref(controller)
    for name,module in mambas:
        if module.activation not in ('silu','swish'):raise ValueError('Expected SiLU Mamba activation.')
        def forward(self,x,inference_params=None,_name=name,_ref=ref):
            return _mamba_forward(self,x,_ref(),_name,inference_params)
        module.forward=types.MethodType(forward,module)
    for i,block in enumerate(blocks):
        prefix=f'mamba_blocks.{i}'
        def bidirectional(self,x,inference_params=None,_prefix=prefix,_ref=ref):
            q=_ref().quant
            fwd=self.mamba_fwd(x,inference_params=inference_params)
            rev_input=q(_prefix+'.reverse_input',x.flip(1))
            rev=self.mamba_rev(rev_input,inference_params=inference_params).flip(1)
            rev=q(_prefix+'.reverse_output',rev)
            return q(_prefix+'.sum',q(_prefix+'.equalize',fwd)+q(_prefix+'.equalize',rev))
        block.forward=types.MethodType(bidirectional,block)
    def classify(self,x,_ref=ref):
        q=_ref().quant
        x=q('classifier.fc1',self.fc1(x))
        x=q('classifier.gelu',self.activation1(x))
        x=self.mamba_1(x)
        x=q('classifier.pool',x.permute(0,2,1).contiguous().mean(-1))
        return self.fc3(x)
    head.forward=types.MethodType(classify,head)
    def fem_forward(self,x,mask,_ref=ref):
        q=_ref().quant;original=x
        masked=x.clone();masked[mask]=0
        x=q('patch',self.patch_embed(q('input',masked)))
        position=q('position',self.pos_embed)
        x=q('embed_add',q('embed_add',x)+q('embed_add',position))
        for i,(block,norm) in enumerate(zip(self.mamba_blocks,self.norm_layers)):
            x=q(f'norm_layers.{i}',norm(x+block(x)))
        return self.classifier(x),original
    model.forward=types.MethodType(fem_forward,model)
    controller.to(next(model.parameters()).device)
    controller.set_enabled(False)
    return controller

def calibrate(model,controller,batches):
    from brevitas.graph.calibrate import calibration_mode
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
    if any(s.counter!=0 for s in controller.observers().values()):raise ValueError('Load fresh activation observers.')
    controller.set_enabled(True)
    proxies={n:p for n,p in controller.named_modules() if isinstance(p,ActQuantProxyFromInjectorBase) and p.is_quant_enabled}
    visits=dict.fromkeys(proxies,0)
    def visit(name):visits[name]+=1
    hooks=[p.register_forward_pre_hook(lambda m,a,_n=n:visit(_n)) for n,p in proxies.items()]
    count=0
    try:
        with torch.no_grad(),calibration_mode(model):
            for x in batches:
                y=model(x,torch.zeros_like(x,dtype=torch.bool))[0]
                if not bool(torch.isfinite(y).all()):raise RuntimeError('Nonfinite calibration logits.')
                count+=1
    finally:
        for hook in hooks:hook.remove()
        model.eval()
    if not count or any(n==0 for n in visits.values()):raise RuntimeError('Calibration missed activation points.')
    return {'num_batches':count,'forward_batches':count,'stages':1,'activation_quantizers':len(controller.names),
            'points':controller.names,'visits':visits,'scales':{n:float(v) for n,v in controller.frozen_scales().items()}}
