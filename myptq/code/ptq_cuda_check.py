"""CUDA smoke check for the same functional options as ptq.py; no dataset accuracy."""
from dataclasses import replace
import json
import sys
import torch
import ptq
from ptq_runtime import require_cuda,runtime_record
from ptq_quant import apply_ptq
from ptq_calibration import install_activations,calibrate


def main(argv=None):
    p=ptq.parser();p.description=__doc__
    argv=list(sys.argv[1:] if argv is None else argv)
    args=p.parse_args(argv)
    args._explicit={a.split('=',1)[0] for a in argv if a.startswith('--')}
    if args.load_calibrated:p.error('Use the main evaluation entry to load a frozen checkpoint.')
    cfg=ptq.configuration(args);require_cuda(args.device)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model_config,norm,state=ptq.checkpoint_spec(payload,args.repo.resolve(),args.experiment)
        model=ptq.build_model(model_config,state,args.repo.resolve())
    else:
        sys.path.insert(0, str(args.repo.expanduser().resolve()))
        from models.FEMBA import FEMBA
        model=FEMBA(seq_length=1280,num_channels=22,num_classes=2,embed_dim=35,num_blocks=2,exp=4).float().eval()
        norm={'enabled':True,'lower':-20,'upper':20}
    original=ptq.tensor_state(model)
    x=ptq.normalize(torch.randn(2,model.num_channels,model.seq_length,device=args.device)*5,norm)
    model.to(args.device).eval()
    with torch.inference_mode():baseline=ptq.logits_for(model,x).clone()
    controller=install_activations(model,cfg);model.cpu()
    apply_ptq(model,replace(cfg,bits=0,classifier_bits=0 if cfg.classifier_bits is not None else None));model.to(args.device)
    with torch.inference_mode():exposed=ptq.logits_for(model,x)
    torch.testing.assert_close(exposed,baseline,rtol=3e-4,atol=1e-4)
    ptq.restore_tensors(model,original);model.cpu();weights=apply_ptq(model,cfg);model.to(args.device).eval()
    effective=ptq.tensor_state(model,original);report=None
    if controller:
        if cfg.calibration_weight_state=='float':ptq.restore_tensors(model,original)
        try:report=calibrate(model,controller,cfg,lambda:iter([x]))
        finally:ptq.restore_tensors(model,effective)
    with torch.inference_mode():value=ptq.logits_for(model,x)
    if not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite smoke output.')
    ptq.assert_tensors(model,effective)
    torch.cuda.synchronize()
    result={'status':'passed','synthetic_smoke_only':True,'synthetic_windows':2,'config':cfg.to_dict(),
            'runtime':runtime_record(),'float_equivalence_max_abs':float((exposed-baseline).abs().max()),
            'quantized_weights':weights['quantized_weight_tensors'],'activation_quantizers':len(controller.names) if controller else 0,
            'calibration_stages':report['stages'] if report else 0,'gpu':torch.cuda.get_device_name()}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':main()
