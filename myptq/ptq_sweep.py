"""在验证集扫描位宽、分组、旋转与校准组合，保存汇总并支持续跑。"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from ptq import group_value,quantization_value,QUANT_FORMATS,write_json
from ptq_quant import PTQConfig
from ptq_runtime import sha_file,file_signature


def grid(args):
    """位宽组合、分组、旋转和校准的笛卡尔积；无效维度按配置去重。"""
    jobs, seen = [], set()
    if args.granularity == 'tensor' and args.group_sizes != [None]:
        raise ValueError('Tensor granularity requires group-sizes none.')
    formats = ([QUANT_FORMATS[name] for name in args.quant_formats]
               if getattr(args, 'quant_formats', None) else
               [(args.bits, a) for a in getattr(args, 'activation_bit_widths', [32])])
    for (bits, activation), group, method in itertools.product(formats, args.group_sizes, args.methods):
        transforms = args.rotation_transforms if method != 'symmetric' else ['qr']
        sizes = args.rotation_sizes if method == 'block' else [64]
        seeds = args.rotation_seeds if method != 'symmetric' else [2025]
        for transform, size, seed, pct, initialization, cal_method, act_pct in itertools.product(
                transforms, sizes, seeds, args.clip_percentiles, getattr(args, 'weight_inits', ['max']),
                getattr(args, 'calibration_methods', ['observer']), getattr(args, 'activation_percentiles', [99.999])):
            sequential = cal_method == 'sequential'
            config = PTQConfig(bits=bits, group_size=group, granularity=args.granularity, method=method,
                rotation_transform=transform, rotation_size=size, rotation_seed=seed, clip_percentile=pct,
                scope=args.scope, classifier_bits=args.classifier_bits, include=tuple(getattr(args, 'include', [])),
                exclude=tuple(getattr(args, 'exclude', [])), weight_init=initialization,
                activation_bits=32 if activation == 0 else activation, calibration_method=cal_method,
                activation_percentile=act_pct,
                calibration_windows=getattr(args, 'calibration_windows', None) or (1024 if sequential else 512),
                calibration_batch_size=getattr(args, 'calibration_batch_size', None) or (128 if sequential else 32),
                calibration_seed=getattr(args, 'calibration_seed', None) if getattr(args, 'calibration_seed', None) is not None else (0 if sequential else 42),
                calibration_weight_state=getattr(args, 'calibration_weight_state', None) or ('quantized' if sequential else 'float')).validate()
            key = json.dumps(config.to_dict(), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            digest = hashlib.sha256(key.encode()).hexdigest()[:10]
            name = f'w{bits}a{config.activation_bits}_{method}_{initialization}_{digest}'
            output = args.output_dir / name
            method_name = {'symmetric': 'symmetric', 'block': 'partial-rotation', 'full': 'full-rotation'}[method]
            command = [sys.executable, str(Path(__file__).with_name('ptq.py')), '--repo', str(args.repo),
                '--checkpoint', str(args.checkpoint), '--output-dir', str(output), '--split', 'val',
                '--weight-bits', str(bits), '--group-size', str(group) if group is not None else 'none',
                '--granularity', args.granularity, '--quantization', method_name, '--weight-init', initialization,
                '--clip-percentile', str(pct), '--weight-scope', args.scope, '--activation-bits', str(config.activation_bits),
                '--device', args.device, '--batch-size', str(args.batch_size), '--max-batches', str(args.max_batches),
                '--experiment', args.experiment]
            if method != 'symmetric':
                command += ['--rotation-transform', transform, '--rotation-seed', str(seed)]
            if method == 'block':
                command += ['--rotation-block-size', str(size)]
            if args.classifier_bits is not None:
                command += ['--classifier-bits', str(args.classifier_bits)]
            for option in ('include', 'exclude'):
                if getattr(args, option, []):
                    command += ['--' + option] + getattr(args, option)
            for option in ('data_dir', 'data_file'):
                if getattr(args, option, None) is not None:
                    command += ['--' + option.replace('_', '-'), str(getattr(args, option))]
            if config.activation_bits != 32:
                command += ['--calibration-method', cal_method, '--activation-percentile', str(act_pct),
                    '--calibration-windows', str(config.calibration_windows), '--calibration-batch-size', str(config.calibration_batch_size),
                    '--calibration-seed', str(config.calibration_seed), '--calibration-weight-state', config.calibration_weight_state]
                if getattr(args, 'calibration_file', None):
                    command += ['--calibration-file', str(args.calibration_file)]
            jobs.append({'name': name, 'config': config.to_dict(), 'argv': command, 'output': str(output)})
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--data-file', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--quant-formats', choices=QUANT_FORMATS, nargs='+')
    p.add_argument('--weight-bits', '--bits', dest='bits', type=int, choices=[4, 8], default=4)
    p.add_argument('--activation-bit-widths', type=int, choices=[0, 4, 8, 32], nargs='+', default=[32])
    p.add_argument('--group-sizes', type=group_value, nargs='+', default=[32, 64, 128, None])
    p.add_argument('--granularity', choices=['channel', 'tensor'], default='channel')
    p.add_argument('--quantizations', '--methods', dest='methods', type=quantization_value, nargs='+',
                   default=['symmetric', 'block', 'full'])
    p.add_argument('--rotation-transforms', choices=['qr', 'hadamard'], nargs='+', default=['qr', 'hadamard'])
    p.add_argument('--rotation-block-sizes', '--rotation-sizes', dest='rotation_sizes', type=int,
                   choices=[32, 64, 128], nargs='+', default=[32, 64, 128])
    p.add_argument('--rotation-seeds', type=int, nargs='+', default=[2025])
    p.add_argument('--clip-percentiles', type=float, nargs='+', default=[100.])
    p.add_argument('--weight-inits', choices=['max', 'mse'], nargs='+', default=['max'])
    p.add_argument('--weight-scope', '--scope', dest='scope',
                   choices=['encoder-linear', 'all-linear', 'all-weights'], default='encoder-linear')
    p.add_argument('--classifier-bits', type=int, choices=[0, 4, 8])
    p.add_argument('--include', nargs='+', default=[])
    p.add_argument('--exclude', nargs='+', default=[])
    p.add_argument('--calibration-methods', choices=['observer', 'sequential'], nargs='+', default=['observer'])
    p.add_argument('--activation-percentiles', type=float, nargs='+', default=[99.999])
    p.add_argument('--calibration-windows', type=int)
    p.add_argument('--calibration-batch-size', type=int)
    p.add_argument('--calibration-seed', type=int)
    p.add_argument('--calibration-weight-state', choices=['float', 'quantized'])
    p.add_argument('--calibration-file', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--experiment', default='FEMBA_tiny_finetune')
    p.add_argument('--run', action='store_true', help='执行扫描；默认只生成计划')
    args = p.parse_args(argv)
    if args.data_dir is None and args.data_file is None:
        p.error('Supply data-dir or data-file.')
    if args.batch_size <= 0 or args.max_batches < 0:
        p.error('Invalid batch size / max-batches.')
    for name in ('calibration_windows', 'calibration_batch_size'):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            p.error('Calibration windows/batch must be positive.')
    if args.calibration_seed is not None and args.calibration_seed < 0:
        p.error('Calibration seed must be nonnegative.')
    for name in ('repo', 'checkpoint', 'data_dir', 'data_file', 'calibration_file', 'output_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    jobs = grid(args)
    active = any(j['config']['activation_bits'] != 32 for j in jobs)
    if active and args.data_dir is None and args.calibration_file is None:
        p.error('Activation calibration needs training data.')
    plan = {'cases': jobs, 'cases_count': len(jobs), 'split': 'val', 'weight_only': not active}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.run:
        write_json(args.output_dir / 'myptq_plan.json', plan)
        print(f'{len(jobs)} validation cases.')
        return plan

    data = (args.data_file or args.data_dir / 'val.h5').resolve()
    campaign = {'plan': plan, 'checkpoint_sha256': sha_file(args.checkpoint),
                'evaluation_data': file_signature(data), 'calibration_data': None}
    if active:
        train = (args.calibration_file or args.data_dir / 'train.h5').resolve()
        if train == data:
            raise ValueError('Calibration and evaluation data must differ.')
        campaign['calibration_data'] = file_signature(train)
    manifest = args.output_dir / 'myptq_campaign.json'
    # 续跑只比较父模型、数据和实验设置；每次运行的代码／环境另记在各自元数据中。
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != campaign:
        raise ValueError('Sweep data/settings changed; use a new directory.')
    write_json(manifest, campaign)
    write_json(args.output_dir / 'myptq_plan.json', plan)
    rows = []
    for job in jobs:
        root = Path(job['output'])
        receipt = root / 'myptq_complete.json'
        if not receipt.exists():
            with (args.output_dir / (job['name'] + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run(job['argv'], cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(receipt.read_text(encoding='utf-8'))
        expected = {'checkpoint_sha256': campaign['checkpoint_sha256'], 'config': job['config'], 'split': 'val',
                    'data_file': str(data), 'data_bytes': campaign['evaluation_data']['bytes'],
                    'data_mtime_ns': campaign['evaluation_data']['mtime_ns'], 'partial_evaluation': bool(args.max_batches),
                    'calibration_data': campaign['calibration_data'] if job['config']['activation_bits'] != 32 else None}
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Completion settings differ: ' + str(receipt))
        metrics = json.loads((root / 'myptq_metrics.json').read_text(encoding='utf-8'))
        rows.append({'case': job['name'], **job['config'], **metrics})
        with (args.output_dir / 'myptq_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not args.max_batches:
        best = max(rows, key=lambda r: (r['auroc'], r['average_precision'], -r['loss']))
        root = args.output_dir / best['case']
        write_json(args.output_dir / 'myptq_best_validation.json', {**best, 'checkpoint_file': str(root / 'myptq.ckpt')})
    return rows


if __name__ == '__main__':
    main()
