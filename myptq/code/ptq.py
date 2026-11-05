"""PTQ fake quant：权重分组／旋转和激活校准。"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from ptq_quant import PTQConfig, apply_ptq
from ptq_runtime import sha_file, file_signature, implementation_record, runtime_record, require_cuda
from ptq_calibration import install_activations, choose_indices, calibrate

FORMAT = 'myptq-unified-v2'
QUANT_FORMATS = {'W8A32': (8, 32), 'W4A32': (4, 32), 'W8A8': (8, 8),
                 'W4A8': (4, 8), 'W4A4': (4, 4)}
QUANT_FLAGS = {'--bits': 'bits', '--weight-bits': 'bits', '--group-size': 'group_size',
    '--granularity': 'granularity', '--method': 'method', '--quantization': 'method',
    '--rotation-transform': 'rotation_transform', '--rotation-size': 'rotation_size',
    '--rotation-block-size': 'rotation_size', '--rotation-seed': 'rotation_seed',
    '--clip-percentile': 'clip_percentile', '--scope': 'scope', '--weight-scope': 'scope',
    '--classifier-bits': 'classifier_bits', '--include': 'include', '--exclude': 'exclude',
    '--weight-init': 'weight_init', '--activation-bits': 'activation_bits',
    '--calibration-method': 'calibration_method', '--activation-percentile': 'activation_percentile',
    '--calibration-windows': 'calibration_windows', '--calibration-batch-size': 'calibration_batch_size',
    '--calibration-seed': 'calibration_seed', '--calibration-weight-state': 'calibration_weight_state'}


def group_value(value):
    if str(value).lower() in ('none', '0'):
        return None
    if int(value) not in (32, 64, 128):
        raise argparse.ArgumentTypeError('Use 32/64/128/none.')
    return int(value)


def quantization_value(value):
    method = {'partial-rotation': 'block', 'full-rotation': 'full'}.get(value, value)
    if method not in ('symmetric', 'block', 'full'):
        raise argparse.ArgumentTypeError('Use symmetric / partial-rotation / full-rotation.')
    return method


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data = p.add_argument_group('输入与输出')
    data.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    data.add_argument('--checkpoint', type=Path, default=os.getenv('CKPT_PATH'), help='微调后的浮点 checkpoint')
    data.add_argument('--data-dir', type=Path, default=os.getenv('DATA_PATH'), help='含 train.h5 / val.h5 / test.h5 的目录')
    data.add_argument('--data-file', type=Path, help='单独指定评估 HDF5')
    data.add_argument('--split', choices=['val', 'test'], default='val')
    data.add_argument('--output-dir', type=Path)
    weight = p.add_argument_group('权重量化')
    weight.add_argument('--quant-format', choices=QUANT_FORMATS, help='同时指定 W 和 A；默认 W4A32')
    weight.add_argument('--weight-bits', '--bits', dest='bits', type=int, choices=[0, 4, 8], default=4,
                        help='独立指定权重位宽；0 表示保持浮点')
    weight.add_argument('--group-size', type=group_value, default=64, help='32 / 64 / 128 / none；默认 64')
    weight.add_argument('--granularity', choices=['channel', 'tensor'], default='channel',
                        help='none 分组时逐输出通道或整层一个尺度')
    weight.add_argument('--quantization', '--method', dest='method', type=quantization_value,
                        default='symmetric', metavar='{symmetric,partial-rotation,full-rotation}')
    weight.add_argument('--rotation-transform', choices=['qr', 'hadamard'], default='qr')
    weight.add_argument('--rotation-block-size', '--rotation-size', dest='rotation_size', type=int,
                        choices=[32, 64, 128], help='部分旋转的块大小，默认 64；独立于 group-size')
    weight.add_argument('--rotation-seed', type=int, default=2025)
    weight.add_argument('--weight-init', choices=['max', 'mse'], default='max', help='absmax 范围或 MSE 搜索范围')
    weight.add_argument('--clip-percentile', type=float, default=100.)
    weight.add_argument('--weight-scope', '--scope', dest='scope',
                        choices=['encoder-linear', 'all-linear', 'all-weights'], default='encoder-linear',
                        help='默认仅编码器 Linear；all-weights 包含分类头与卷积')
    weight.add_argument('--classifier-bits', type=int, choices=[0, 4, 8])
    weight.add_argument('--include', nargs='+', default=[])
    weight.add_argument('--exclude', nargs='+', default=[])
    activation = p.add_argument_group('激活量化与训练集校准')
    activation.add_argument('--activation-bits', type=int, choices=[0, 4, 8, 32], default=32,
                            help='4 / 8 / 32；32 表示不量化激活')
    activation.add_argument('--calibration-method', choices=['observer', 'sequential'], default='observer')
    activation.add_argument('--activation-percentile', type=float, default=99.999)
    activation.add_argument('--calibration-windows', type=int)
    activation.add_argument('--calibration-batch-size', type=int)
    activation.add_argument('--calibration-seed', type=int)
    activation.add_argument('--calibration-weight-state', choices=['float', 'quantized'])
    activation.add_argument('--calibration-file', type=Path, help='默认 data-dir/train.h5')
    run = p.add_argument_group('运行')
    run.add_argument('--load-calibrated', action='store_true', help='加载已保存的 myptq.ckpt，不重新量化或校准')
    run.add_argument('--device', default='cuda')
    run.add_argument('--batch-size', type=int, default=64)
    run.add_argument('--max-batches', type=int, default=0, help='0 表示完整评估')
    run.add_argument('--experiment', default='FEMBA_tiny_finetune')
    run.add_argument('--print-config', action='store_true', help='仅查看解析后的量化设置')
    return p


def configuration(args):
    bits, activation_bits = args.bits, 32 if args.activation_bits == 0 else args.activation_bits
    if args.quant_format:
        bits, activation_bits = QUANT_FORMATS[args.quant_format]
        for flag, field, expected in [('--weight-bits', 'bits', bits), ('--bits', 'bits', bits),
                                      ('--activation-bits', 'activation_bits', activation_bits)]:
            provided = getattr(args, field)
            if field == 'activation_bits' and provided == 0:
                provided = 32
            if flag in getattr(args, '_explicit', set()) and provided != expected:
                raise ValueError('quant-format conflicts with ' + flag)
    if args.method != 'block' and args.rotation_size is not None:
        raise ValueError('rotation-block-size applies only to partial-rotation.')
    sequential = args.calibration_method == 'sequential'
    return PTQConfig(bits=bits, group_size=args.group_size, granularity=args.granularity, method=args.method,
        rotation_transform=args.rotation_transform, rotation_size=args.rotation_size or 64, rotation_seed=args.rotation_seed,
        clip_percentile=args.clip_percentile, scope=args.scope, classifier_bits=args.classifier_bits,
        include=tuple(args.include), exclude=tuple(args.exclude), weight_init=args.weight_init,
        activation_bits=activation_bits, calibration_method=args.calibration_method,
        activation_percentile=args.activation_percentile,
        calibration_windows=args.calibration_windows if args.calibration_windows is not None else (1024 if sequential else 512),
        calibration_batch_size=args.calibration_batch_size if args.calibration_batch_size is not None else (128 if sequential else 32),
        calibration_seed=args.calibration_seed if args.calibration_seed is not None else (0 if sequential else 42),
        calibration_weight_state=args.calibration_weight_state or ('quantized' if sequential else 'float')).validate()


def config_from_dict(value):
    value = dict(value)
    value['include'], value['exclude'] = tuple(value.get('include', [])), tuple(value.get('exclude', []))
    defaults = PTQConfig()
    for name in ('rotation_transform', 'rotation_size', 'rotation_seed', 'calibration_method', 'activation_percentile',
                 'calibration_windows', 'calibration_batch_size', 'calibration_seed', 'calibration_weight_state'):
        if value.get(name) is None:
            value[name] = getattr(defaults, name)
    return PTQConfig(**value).validate()


def saved_configuration(payload, args):
    if payload.get('format') not in ('myptq-unified-v1', FORMAT):
        raise ValueError('Use a myptq checkpoint with --load-calibrated.')
    config = config_from_dict(payload['config'])
    canonical = config.to_dict()
    if args.quant_format and QUANT_FORMATS[args.quant_format] != (config.bits, config.activation_bits):
        raise ValueError('quant-format conflicts with frozen checkpoint.')
    # 已冻结的数值不能通过命令行改位宽；不检查当前配置中无效的选项。
    for flag in getattr(args, '_explicit', set()) & set(QUANT_FLAGS):
        field = QUANT_FLAGS[flag]
        if canonical[field] is None and (field.startswith(('rotation_', 'calibration_')) or field == 'activation_percentile'):
            continue
        provided = getattr(args, field)
        if field in ('include', 'exclude'):
            provided = list(provided)
        if field == 'activation_bits' and provided == 0:
            provided = 32
        if canonical[field] != provided:
            raise ValueError('Explicit setting conflicts with frozen checkpoint: ' + flag)
    return config


def source_training_precision(payload):
    """AMP checkpoint 的主权重通常仍是 FP32；训练精度应读配置。"""
    if 'finetune_precision' in payload:
        return payload['finetune_precision']
    trainer = (payload.get('hyper_parameters') or {}).get('trainer') or {}
    value = trainer.get('precision')
    return {'32': 'fp32', '32-true': 'fp32', '16': 'fp16', '16-mixed': 'fp16',
            'bf16': 'bf16', 'bf16-mixed': 'bf16'}.get(str(value), str(value) if value is not None else 'not_recorded')


def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def tensor_state(model,names=None):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if names is None or n in names}


@torch.no_grad()
def restore_tensors(model,values):
    state=model.state_dict()
    for name,value in values.items():state[name].copy_(value.to(state[name].device))


def assert_tensors(model,values):
    state=model.state_dict()
    if any(not torch.equal(v,state[n].detach().cpu()) for n,v in values.items()):raise RuntimeError('Cached model tensors changed during calibration/evaluation.')


def checkpoint_spec(payload, repo, experiment):
    """Support Lightning classifier checkpoints and run_femba_baseline float checkpoints."""
    from omegaconf import OmegaConf
    if payload.get('stage') == 'ptq' or payload.get('method') == 'PTQ':
        raise ValueError('Each PTQ case must start from the original FLOAT checkpoint.')
    state = payload.get('state_dict', payload)
    if any('.parametrizations.weight.' in k or k.startswith('model.act_quant.') for k in state):
        raise ValueError('QAT/QAP checkpoint detected; use the original floating-point classifier.')
    if 'model_config' in payload:
        model_config = dict(payload['model_config'])
        norm = dict(payload.get('normalization', {}))
        normalization = {'enabled': norm.get('enabled', norm.get('normalize', True)),
                         'lower': float(norm.get('lower', -20)), 'upper': float(norm.get('upper', 20))}
    else:
        if not OmegaConf.has_resolver('env'):
            OmegaConf.register_new_resolver('env', lambda name: os.getenv(name))
        if 'hyper_parameters' in payload:
            cfg = OmegaConf.create(payload['hyper_parameters'])
        else:
            from hydra import compose, initialize_config_dir
            with initialize_config_dir(config_dir=str(repo/'config'), version_base='1.3'):
                cfg = compose(config_name='defaults', overrides=['+experiment='+experiment])
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        norm = cfg.get('input_normalization')
        normalization = {'enabled': bool(norm is not None and norm.get('normalize', False)),
                         'lower': float(norm.get('quartile_normalization_lower_val', -20)) if norm is not None else -20.,
                         'upper': float(norm.get('quartile_normalization_upper_val', 20)) if norm is not None else 20.}
    target = model_config.pop('_target_', 'models.FEMBA.FEMBA')
    if target != 'models.FEMBA.FEMBA':
        raise ValueError(f'This entry targets the original FEMBA, not {target}.')
    if int(model_config.get('num_classes', 0)) < 2:
        raise ValueError('A fine-tuned classification checkpoint is required, not a pretraining decoder.')
    if model_config.get('classification_type', 'bc') not in ('bc','mcc','ml'):
        raise ValueError('Only single-label classification is supported by this evaluator.')
    if normalization['upper'] <= normalization['lower']:
        raise ValueError('Invalid normalization bounds.')
    if any(k.startswith('model.') for k in state):
        state = {k[len('model.'):]:v for k,v in state.items() if k.startswith('model.')}
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint does not contain a usable model state_dict.')
    return model_config, normalization, state


def build_model(model_config, state, repo):
    sys.path.insert(0, str(repo))
    from models.FEMBA import FEMBA
    model = FEMBA(**model_config).cpu().float().eval()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def build_dataset(path, repo):
    # Load the exact local dataset file, avoiding conflicts with the HuggingFace datasets package.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_myptq_hdf5', repo/'datasets/hdf5_dataset.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.HDF5Loader(str(path), finetune=True, use_cache=False)


def normalize(x, settings):
    if settings['enabled']:
        return (x - settings['lower']) / (settings['upper'] - settings['lower'] + 1e-8)
    return x


def logits_for(model, x):
    output = model(x, torch.zeros_like(x, dtype=torch.bool))
    return output[0] if isinstance(output, (tuple, list)) else output


def compute_metrics(logits, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or values.ndim != 2 or len(labels) != len(values) or not len(labels):
        raise ValueError('Expected nonempty logits [N,C] and single class labels [N].')
    if not np.isfinite(values).all() or not np.isfinite(labels).all() or not np.equal(labels, labels.astype(np.int64)).all():
        raise ValueError('Non-finite outputs or noninteger labels.')
    labels = labels.astype(np.int64)
    count = values.shape[1]
    if set(labels.tolist()) != set(range(count)):
        raise ValueError('The evaluated subset must contain every class for AUROC/AP.')
    tensor = torch.from_numpy(values)
    probs = tensor.softmax(1).numpy()
    if count == 2:
        scores = values[:,1] - values[:,0]
        predicted = (scores >= 0).astype(np.int64)
        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        f1 = f1_score(labels, predicted, zero_division=0)
        definition = 'float64(z1)-float64(z0); threshold=0'
    else:
        predicted = probs.argmax(1)
        auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        ap = average_precision_score(np.eye(count)[labels], probs, average='macro')
        f1 = f1_score(labels, predicted, average='macro', zero_division=0)
        definition = 'float64_softmax; macro_ovr'
    confidence = probs[np.arange(len(labels)), predicted]
    correct = predicted == labels
    ece = 0.
    for index in range(15):
        selected = (confidence > index/15) & (confidence <= (index+1)/15)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    result = {'samples':len(labels), 'auroc':float(auroc), 'average_precision':float(ap),
              'accuracy':float(correct.mean()), 'balanced_accuracy':float(balanced_accuracy_score(labels,predicted)),
              'f1':float(f1), 'ece_15_bins':float(ece),
              'loss':float(F.cross_entropy(tensor,torch.from_numpy(labels))), 'score_definition':definition}
    if count == 2:
        result['auroc_legacy_positive_logit'] = float(roc_auc_score(labels,values[:,1]))
    return result


@torch.inference_mode()
def evaluate(model, loader, normalization, device, max_batches=0):
    all_logits, all_labels = [], []
    for index, (x,y) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        outputs = logits_for(model, normalize(x.to(device).float(), normalization))
        if not bool(torch.isfinite(outputs).all()):
            raise ValueError('Non-finite model logits; no valid precision result can be reported.')
        all_logits.append(outputs.float().cpu().numpy()); all_labels.append(y.numpy().reshape(-1))
    if not all_logits:
        raise ValueError('Evaluation dataset is empty.')
    logits, labels = np.concatenate(all_logits), np.concatenate(all_labels)
    return compute_metrics(logits,labels), logits, labels


def execute(args, config=None):
    if args.device.startswith('cuda'):
        require_cuda(args.device)
    repo, checkpoint = args.repo.expanduser().resolve(), args.checkpoint.expanduser().resolve()
    data = (args.data_file or args.data_dir / f'{args.split}.h5').expanduser().resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    parent_hash = sha_file(checkpoint)
    finetune_precision = source_training_precision(payload)
    if args.load_calibrated:
        config = saved_configuration(payload, args)
        model_config, norm, state = payload['model_config'], payload['normalization'], None
        source_dtypes = payload.get('source_weight_dtypes', [])
    else:
        config = config or configuration(args)
        model_config, norm, state = checkpoint_spec(payload, repo, args.experiment)
        source_dtypes = sorted({str(t.dtype) for t in state.values() if t.is_floating_point()})
    active = config.activation_bits != 32
    if args.calibration_file and (args.load_calibrated or not active):
        raise ValueError('calibration-file requires fresh activation calibration.')
    output = (args.output_dir or repo / 'results/myptq' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory: ' + str(output))
    model = build_model(model_config, state, repo)
    original = tensor_state(model) if not args.load_calibrated else None
    output.mkdir(parents=True, exist_ok=True)
    implementation = implementation_record(repo)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset = training = None
    try:
        dataset = build_dataset(data, repo)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.to(args.device).eval()
        check_forward = not args.load_calibrated and (config.method != 'symmetric' or active)
        if check_forward:
            probe = normalize(next(iter(loader))[0][:2].to(args.device).float(), norm)
            with torch.inference_mode():
                baseline = logits_for(model, probe).clone()
        controller = install_activations(model, config)
        model.to(args.device).eval()
        equivalence = {'checked': False}
        # 开启量化前检查旋转往返与替换后的前向，防止结构改写本身改变模型。
        if check_forward:
            if config.method != 'symmetric':
                model.cpu()
                apply_ptq(model, replace(config, bits=0, classifier_bits=0 if config.classifier_bits is not None else None))
                model.to(args.device)
            with torch.inference_mode():
                exposed = logits_for(model, probe)
            torch.testing.assert_close(exposed, baseline, rtol=3e-4, atol=1e-4)
            equivalence = {'checked': True, 'max_abs_error': float((exposed - baseline).abs().max())}
            restore_tensors(model, original)

        calibration_data = calibration_report = recipe = export_hash = None
        if args.load_calibrated:
            model.load_state_dict(payload['state_dict'], strict=True)
            if controller:
                controller.set_enabled(True)
            weight_report, recipe = payload['weight_report'], payload.get('recipe')
            export_hash = parent_hash
        else:
            # 每个候选从浮点父模型开始，缓存 Q(W R) R^T；推理不再重复量化权重。
            model.cpu()
            weight_report = apply_ptq(model, config)
            model.to(args.device).eval()
            if active:
                train = (args.calibration_file or args.data_dir / 'train.h5').expanduser().resolve()
                if train == data:
                    raise ValueError('Training calibration data must differ from evaluation data.')
                calibration_data = file_signature(train)
                training = build_dataset(train, repo)
                indices = choose_indices(len(training), config)
                write_json(output / 'myptq_calibration_indices.json',
                           {'split': 'train', 'indices': indices, 'seed': config.calibration_seed, 'data': calibration_data})

                def batches():
                    samples = DataLoader(Subset(training, indices), batch_size=config.calibration_batch_size,
                                         shuffle=False, num_workers=0)
                    for x, _ in samples:
                        yield normalize(x.to(args.device).float(), norm)

                # float 校准暂存量化权重，统计结束后恢复；sequential 默认直接用量化权重。
                if config.calibration_weight_state == 'float':
                    effective = tensor_state(model, original)
                    restore_tensors(model, original)
                    try:
                        calibration_report = calibrate(model, controller, config, batches)
                    finally:
                        restore_tensors(model, effective)
                else:
                    calibration_report = calibrate(model, controller, config, batches)
                recipe = {'method': config.calibration_method, 'weight_state': config.calibration_weight_state,
                          'requested_windows': config.calibration_windows, 'actual_windows': len(indices),
                          'batch_size': config.calibration_batch_size, 'seed': config.calibration_seed}
            if weight_report['quantized_weight_tensors'] or active:
                artifact = output / 'myptq.ckpt'
                # 只保存推理需要的有效权重和冻结尺度，不重复保存整份浮点父模型。
                torch.save({'format': FORMAT, 'config': config.to_dict(), 'model_config': model_config,
                            'normalization': norm, 'state_dict': model.state_dict(),
                            'activation_scales': controller.frozen_scales() if controller else {},
                            'weight_report': weight_report, 'recipe': recipe, 'parent_sha256': parent_hash,
                            'implementation': implementation, 'finetune_precision': finetune_precision,
                            'source_weight_dtypes': source_dtypes, 'stage': 'ptq'}, artifact)
                export_hash = sha_file(artifact)
        metrics, logits, labels = evaluate(model, loader, norm, args.device, args.max_batches)
        metadata = {'method': 'PTQ' if weight_report['quantized_weight_tensors'] or active else 'FLOAT_DIAGNOSTIC',
                    'quant_format': f'W{config.bits or 32}A{config.activation_bits}',
                    'finetune_precision': finetune_precision, 'source_weight_dtypes': source_dtypes,
                    'ptq_compute_dtype': 'fp32', 'checkpoint': str(checkpoint), 'checkpoint_sha256': parent_hash,
                    'config': config.to_dict(), 'ptq': weight_report, 'model_config': model_config,
                    'normalization': norm, 'split': args.split, 'data_file': str(data),
                    'partial_evaluation': bool(args.max_batches), 'rotation_equivalence': equivalence,
                    'activation_bits': config.activation_bits, 'activation_quantized': active,
                    'activation_quantizers': len(controller.names) if controller else 0,
                    'calibration_data': calibration_data, 'calibration_report': calibration_report,
                    'recipe': recipe, 'runtime': runtime_record(), 'implementation': implementation,
                    'exported_checkpoint_sha256': export_hash, 'strict_load': True,
                    'execution': 'FP32 fake quant; cached effective weights; CUDA selective scan'}
        np.savez_compressed(output / 'myptq_predictions.npz', logits=logits, labels=labels)
        write_json(output / 'myptq_metadata.json', metadata)
        write_json(output / 'myptq_metrics.json', metrics)
        signature = file_signature(data)
        write_json(output / 'myptq_complete.json',
                   {'checkpoint_sha256': parent_hash, 'config': config.to_dict(), 'split': args.split,
                    'data_file': str(data), 'data_bytes': signature['bytes'], 'data_mtime_ns': signature['mtime_ns'],
                    'partial_evaluation': bool(args.max_batches), 'calibration_data': calibration_data,
                    'exported_checkpoint_sha256': export_hash})
        print(json.dumps({'output': str(output), 'config': config.to_dict(), 'metrics': metrics}, ensure_ascii=False))
        return metrics
    finally:
        for ds in (dataset, training):
            if ds is not None and hasattr(ds, 'data'):
                ds.data.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    args._explicit = {a.split('=', 1)[0] for a in argv if a.startswith('--')}
    if args.batch_size <= 0 or args.max_batches < 0:
        p.error('Invalid batch-size/max-batches.')
    config = None if args.load_calibrated else configuration(args)
    if args.print_config:
        if args.load_calibrated:
            config = saved_configuration(torch.load(args.checkpoint, map_location='cpu', weights_only=False), args)
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.checkpoint is None or (args.data_file is None and args.data_dir is None):
        p.error('Supply checkpoint and data.')
    if not args.load_calibrated and config.activation_bits != 32 and args.data_dir is None and args.calibration_file is None:
        p.error('Activation calibration needs --data-dir or --calibration-file.')
    return execute(args, config)


if __name__ == '__main__':
    main()
