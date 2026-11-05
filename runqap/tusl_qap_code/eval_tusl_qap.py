#!/usr/bin/env python3
"""Evaluate a three-class TUSL W4/W8, A32/A8/A4 QAP checkpoint at window and event level."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

LABELS = {0: 'bckg', 1: 'seiz', 2: 'slow'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def auc_ap(labels, score):
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(y.sum()), int((1-y).sum())
    if not pos or not neg:
        raise ValueError('One-vs-rest AUROC requires positive and negative examples.')
    order = np.argsort(score, kind='stable')
    s = score[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1])+1, len(s)]
    starts = np.r_[0, ends[:-1]]
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.repeat((starts+ends+1)/2, ends-starts)
    auc = (ranks[y == 1].sum() - pos*(pos+1)/2)/(pos*neg)
    order = np.argsort(-score, kind='stable')
    s, yy = score[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.cumsum(yy)[ends]
    precision, recall = tp/(ends+1), tp/pos
    ap = np.sum(np.diff(np.r_[0.0, recall])*precision)
    return float(auc), float(ap)


def metrics(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('TUSL requires one integer class per window/event.')
    if probs.shape != (len(labels), 3) or set(labels.tolist()) != {0, 1, 2}:
        raise ValueError('Expected all three classes and probabilities of shape [N, 3].')
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(1), 1):
        raise ValueError('Invalid class probabilities.')
    pred = probs.argmax(1)
    cm = np.bincount(labels*3+pred, minlength=9).reshape(3, 3)
    per_class = {}
    for c, name in LABELS.items():
        tp = int(cm[c, c]); support = int(cm[c].sum()); predicted = int(cm[:, c].sum())
        auc, ap = auc_ap(labels == c, probs[:, c])
        per_class[name] = {'label': c, 'support': support, 'auroc_ovr': auc, 'ap_ovr': ap,
                           'recall': tp/support, 'precision': tp/max(predicted, 1),
                           'f1': 2*tp/max(support+predicted, 1)}
    return {'n': len(labels), 'accuracy': float((pred == labels).mean()),
            'balanced_accuracy': float(np.mean([r['recall'] for r in per_class.values()])),
            'macro_f1': float(np.mean([r['f1'] for r in per_class.values()])),
            'macro_auroc_ovr': float(np.mean([r['auroc_ovr'] for r in per_class.values()])),
            'macro_ap_ovr': float(np.mean([r['ap_ovr'] for r in per_class.values()])),
            'cross_entropy': float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-15, 1)).mean()),
            'confusion_matrix': cm.tolist(), 'class_order': list(LABELS.values()), 'per_class': per_class}


def aggregate_events(probs, labels, names):
    if len(names) != len(set(names)) or not (len(names) == len(labels) == len(probs)):
        raise ValueError('Duplicate or mismatched source_file entries.')
    grouped = {}
    for i, name in enumerate(names):
        match = re.fullmatch(r'(.+)_w([01])', str(name))
        if not match:
            raise ValueError(f'Expected event_w0 / event_w1: {name}')
        grouped.setdefault(match[1], []).append((int(match[2]), i))
    event_names, event_labels, event_probs = [], [], []
    for event, members in sorted(grouped.items()):
        if sorted(w for w, _ in members) != [0, 1]:
            raise ValueError(f'Event must have exactly two windows: {event}')
        index = [i for _, i in members]
        if len(set(int(labels[i]) for i in index)) != 1:
            raise ValueError(f'Conflicting window labels: {event}')
        event_names.append(event); event_labels.append(int(labels[index[0]]))
        event_probs.append(np.asarray(probs)[index].mean(0))
    return np.asarray(event_names), np.asarray(event_labels, dtype=np.int64), np.asarray(event_probs)


def validate_config(cfg, *, pretrain=False, weight_bits=4, act_bits=8, percentile=99.9):
    if int(cfg.get('qap_weight_bits', -1)) != weight_bits:
        raise ValueError('Checkpoint weight bits differ from the requested experiment.')
    suffix = 'pretrain_qap_task.QAPPretrainTask' if pretrain else 'finetune_qap_task.QAPFinetuneTask'
    if not str(cfg['task']['_target_']).endswith(suffix):
        raise ValueError('Checkpoint is from the wrong task stage.')
    model = cfg['model']
    if int(model['num_classes']) != (0 if pretrain else 3):
        raise ValueError('Expected reconstruction pretraining or a three-class classifier.')
    if not pretrain and model['classification_type'] != 'mcc':
        raise ValueError('Expected classification_type=mcc.')
    if (int(model['num_blocks']), int(model['embed_dim'])) != (2, 35):
        raise ValueError('Expected FEMBA-Tiny: two blocks, embed_dim=35.')
    if (int(model['num_channels']), int(model['seq_length'])) != (22, 1280):
        raise ValueError('Expected TUSL input [22, 1280].')
    if int(cfg.get('qap_act_bits', 0)) != act_bits:
        raise ValueError('Checkpoint activation bits differ from the requested experiment.')
    if act_bits and not np.isclose(float(cfg.get('qap_act_percentile', -1)), percentile, rtol=0, atol=1e-9):
        raise ValueError('Checkpoint activation percentile differs from the requested experiment.')



def validate_activation_state(state, *, pretrain=False, act_bits=8):
    """Validate the saved learned activation ranges; never observe or recalibrate test inputs."""
    import torch
    keys=[k for k in state if k.startswith('model.act_quant.q.') and k.endswith('.log_range')]
    expected=(25 if pretrain else 32) if act_bits else 0
    if len(keys)!=expected:
        raise ValueError(f'Expected {expected} saved activation ranges; found {len(keys)}.')
    for key in keys:
        prefix=key[:-len('log_range')]
        if not bool(state[prefix+'initialized']) or not bool(state[prefix+'enabled']):
            raise ValueError(f'Inactive or uninitialized saved activation quantizer: {key}')
        value=state[key]
        r=value.exp()
        if not bool(torch.isfinite(value).all() and torch.isfinite(r).all() and (r>0).all()):
            raise ValueError(f'Invalid learned activation range: {key}')
    return len(keys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--fold-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pretrain-checkpoint', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--expected-weight-bits',type=int,choices=[0,4,8],default=4)
    p.add_argument('--expected-act-bits',type=int,choices=[0,4,8],default=8)
    p.add_argument('--expected-act-percentile',type=float,default=99.9)
    args = p.parse_args()
    for key in ('repo','fold_dir','checkpoint','pretrain_checkpoint','out_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    sys.path.insert(0, str(args.repo))
    os.environ.setdefault('DATA_PATH', str(args.fold_dir))
    os.environ.setdefault('CHECKPOINT_DIR', str(args.out_dir))
    import h5py
    import torch
    from omegaconf import OmegaConf
    if not torch.cuda.is_available():
        raise RuntimeError('Run inference in the CUDA/Mamba environment on a GPU node.')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive.')
    if not OmegaConf.has_resolver('env'):
        OmegaConf.register_new_resolver('env', lambda key: os.environ[key])
    if (args.out_dir/'metrics.json').exists():
        raise FileExistsError('Existing metrics.json: choose a new output directory.')
    data = args.fold_dir/'test.h5'
    manifest = json.loads((args.fold_dir/'data-manifest.json').read_text())
    if manifest['num_classes'] != 3 or manifest['classification_type'] != 'mcc':
        raise ValueError('Wrong TUSL fold manifest.')
    data_hash = sha(data)
    if data_hash != manifest['data']['test']['sha256']:
        raise ValueError('Test HDF5 does not match the fold manifest.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    parent = torch.load(args.pretrain_checkpoint, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(checkpoint['hyper_parameters'])
    parent_cfg = OmegaConf.create(parent['hyper_parameters'])
    validate_config(cfg,weight_bits=args.expected_weight_bits,act_bits=args.expected_act_bits,percentile=args.expected_act_percentile)
    validate_config(parent_cfg,pretrain=True,weight_bits=args.expected_weight_bits,act_bits=args.expected_act_bits,percentile=args.expected_act_percentile)
    n_saved_act=validate_activation_state(checkpoint['state_dict'],act_bits=args.expected_act_bits)
    validate_activation_state(parent['state_dict'],pretrain=True,act_bits=args.expected_act_bits)
    if cfg.seed != parent_cfg.seed:
        raise ValueError('Pretraining and fine-tuning seeds differ.')
    if Path(str(cfg.pretrained_checkpoint_path)).resolve() != args.pretrain_checkpoint:
        raise ValueError('Given parent checkpoint path differs from the recorded fine-tuning input.')
    for config in (cfg, parent_cfg):
        for split in ('train','val'):
            if Path(str(config.data_module[split].hdf5_file)).resolve() != (args.fold_dir/f'{split}.h5').resolve():
                raise ValueError('Checkpoint was configured for a different fold.')
    from tasks.finetune_task import FinetuneTask
    from models.weight_quant import add_weight_quant, check_weight_quant, iter_weight_quantizers
    task = FinetuneTask(cfg)
    n_weights = len(add_weight_quant(task.model, args.expected_weight_bits)) if args.expected_weight_bits else 0
    expected_weights = 28 if args.expected_weight_bits else 0
    if n_weights != expected_weights:
        raise ValueError(f'Expected {expected_weights} quantized weight tensors, got {n_weights}.')
    # Install the exact activation graph BEFORE restoring the learned ranges and enable flags.
    if args.expected_act_bits:
        from models.act_quant import install_act_quant
        aq=install_act_quant(task.model,args.expected_act_bits)
        if len(aq.q)!=n_saved_act:
            raise ValueError('Activation graph does not match saved state.')
    task.load_state_dict(checkpoint['state_dict'], strict=True)
    if args.expected_act_bits:
        if aq.count_enabled()!=n_saved_act:
            raise ValueError('Some activation quantizers are disabled after loading.')
        if any(q.calibrating or q.bits!=args.expected_act_bits for q in aq.q.values()):
            raise ValueError('Inference must use the trained bit width with calibration OFF.')
    for name, module, quant in iter_weight_quantizers(task.model):
        if not bool(quant.enabled) or not bool(quant.initialized) or quant.bits != args.expected_weight_bits:
            raise ValueError(f'Inactive/uninitialized weight quantizer: {name}')
        if not torch.isfinite(module.parametrizations.weight.original).all() or not torch.isfinite(quant.log_scale).all():
            raise ValueError(f'Non-finite quantized weights: {name}')
    _, weight_error = check_weight_quant(task.model) if args.expected_weight_bits else (0, 0.0)
    if not np.isfinite(weight_error):
        raise ValueError('Non-finite weight quantization error.')
    task.eval().cuda()
    logits_list, labels_list, source_names = [], [], []
    with torch.inference_mode(), h5py.File(data, 'r') as f:
        groups=[g for g in f.values() if isinstance(g,h5py.Group)]
        if not groups: raise ValueError('Test HDF5 contains no data groups.')
        if any('source_file' not in group for group in groups):
            raise ValueError('Each test group must contain source_file for event aggregation.')
        for group in groups:
            if group['X'].shape[1:] != (22,1280):
                raise ValueError('Wrong test signal shape.')
            for start in range(0, len(group['X']), args.batch_size):
                stop = start+args.batch_size
                X = torch.as_tensor(group['X'][start:stop], dtype=torch.float32, device='cuda')
                if getattr(task, 'normalize', False): X = task.normalize_fct(X)
                mask = torch.zeros_like(X, dtype=torch.bool)
                logits = task.model(X, mask)[0]
                logits_list.append(logits.float().cpu().numpy())
                labels_list.append(group['y'][start:stop])
                source_names.extend(group['source_file'].asstr()[start:stop].tolist())
    logits = np.concatenate(logits_list).astype(np.float64)
    labels = np.concatenate(labels_list).astype(np.int64)
    if logits.shape != (len(labels),3) or not np.isfinite(logits).all():
        raise ValueError('Expected finite three-class logits.')
    exp = np.exp(logits-logits.max(1, keepdims=True)); probs = exp/exp.sum(1, keepdims=True)
    if set(labels.tolist())!={0,1,2}: raise ValueError('Test split must contain all three classes.')
    names, event_labels, event_probs = aggregate_events(probs, labels, source_names)
    event_metrics=metrics(event_probs,event_labels)
    n_subjects=len({n.split('_',1)[0] for n in source_names})
    aggregation='arithmetic mean of two window softmax probabilities'
    output = {'method': (f'QAP W{args.expected_weight_bits}A{args.expected_act_bits or 32}' if args.expected_weight_bits or args.expected_act_bits else 'FP32 W32A32'), 'fold': manifest['fold'], 'seed': int(cfg.seed),
              'weight_bits':args.expected_weight_bits,'act_bits':args.expected_act_bits,
              'act_percentile':args.expected_act_percentile if args.expected_act_bits else None,
              'n_act_quantizers':n_saved_act,'test_activation_calibration':False,'num_classes':3, 'n_quantized_weight_tensors': n_weights,
              'weight_quant_relative_error': weight_error, 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': sha(args.checkpoint), 'pretrain_checkpoint': str(args.pretrain_checkpoint),
              'pretrain_checkpoint_sha256': sha(args.pretrain_checkpoint), 'test_h5_sha256': data_hash,
              'checkpoint_epoch': int(checkpoint['epoch']), 'checkpoint_global_step': int(checkpoint['global_step']),
              'n_test_subjects': n_subjects, 'source_metadata_available':True,
              'event_aggregation': aggregation, 'windows': metrics(probs, labels), 'events': event_metrics}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir/'predictions.npz', logits=logits, labels=labels,
                        probabilities=probs, source_file=np.asarray(source_names,dtype=str), event=names,
                        event_labels=event_labels, event_probabilities=event_probs)
    (args.out_dir/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


