"""验证位宽组合、混合精度训练配置及 PTQ 保存／重载；CPU 数值夹具不替代 CUDA 检查。"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import ptq
import ptq_finetune as finetune
import ptq_calibration_test as fixtures
import ptq_quant as weights
import ptq_sweep as sweep


class OptionsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CalibrationTests('test_functional_options_and_profile_defaults')
        self.fixture.setUp()

    def test_requested_formats_groups_and_rotation_sizes(self):
        for name, expected in ptq.QUANT_FORMATS.items():
            for group in ('32', '64', '128', 'none'):
                for method in ('symmetric', 'partial-rotation', 'full-rotation'):
                    for size in ((32, 64, 128) if method == 'partial-rotation' else (None,)):
                        argv = ['--quant-format', name, '--group-size', group, '--quantization', method]
                        if size:
                            argv += ['--rotation-block-size', str(size)]
                        cfg = ptq.configuration(ptq.parser().parse_args(argv))
                        self.assertEqual((cfg.bits, cfg.activation_bits), expected)
                        self.assertEqual(cfg.group_size, None if group == 'none' else int(group))

    def test_quant_format_conflicting_bits(self):
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            ptq.main(['--quant-format', 'W4A8', '--weight-bits', '8', '--print-config'])

    def test_finetune_precision_configs_use_original_training(self):
        if not OmegaConf.has_resolver('env'):
            OmegaConf.register_new_resolver('env', os.getenv)
        with tempfile.TemporaryDirectory(prefix='myptq-precision-') as folder:
            root = Path(folder)
            for precision, lightning_mode in finetune.PRECISIONS.items():
                args = finetune.parser().parse_args([
                    '--repo', str(fixtures.REPO), '--pretrained-weights', str(root / 'weights with = sign.safetensors'),
                    '--data-dir', str(root), '--output-dir', str(root / precision), '--precision', precision])
                command = finetune.training_command(args)
                with patch.dict(os.environ, {'DATA_PATH': str(root), 'CHECKPOINT_DIR': str(root)}),\
                        initialize_config_dir(config_dir=str(fixtures.REPO / 'config'), version_base='1.1'):
                    cfg = compose(config_name='defaults', overrides=command[3:])
                    self.assertEqual(cfg.trainer.precision, lightning_mode)
                    self.assertEqual(cfg.model._target_, 'models.FEMBA.FEMBA')
                    self.assertEqual(cfg.trainer.devices, 1)
                    self.assertFalse(cfg.resume)
                    self.assertFalse(cfg.final_test)
                    self.assertEqual(cfg.pretrained_safetensors_path, args.pretrained_weights.as_posix())
                    self.assertEqual(ptq.source_training_precision({'hyper_parameters': cfg}), precision)
                with patch.object(finetune.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                    finetune.main(['--repo', str(fixtures.REPO), '--pretrained-weights', str(args.pretrained_weights),
                        '--data-dir', str(root), '--output-dir', str(root / precision), '--precision', precision, '--dry-run'])
                    run.assert_not_called()
                self.assertFalse((root / precision).exists())

    def test_finetune_launch_records_precision_and_environment(self):
        with tempfile.TemporaryDirectory(prefix='myptq-launch-') as folder:
            root = Path(folder)
            with patch('ptq_runtime.require_cuda'), patch.object(finetune.subprocess, 'run') as run,\
                    contextlib.redirect_stdout(io.StringIO()):
                record = finetune.main(['--repo', str(fixtures.REPO), '--pretrained-weights', str(root / 'femba.safetensors'),
                    '--data-dir', str(root), '--output-dir', str(root / 'run'), '--precision', 'bf16'])
            self.assertEqual(record['trainer_precision'], 'bf16-mixed')
            self.assertEqual(run.call_args.kwargs['env']['DATA_PATH'], str(root.resolve()))
            self.assertEqual(run.call_args.kwargs['env']['NVIDIA_TF32_OVERRIDE'], '0')
            self.assertEqual(json.loads((root / 'run/myptq_finetune.json').read_text())['precision'], 'bf16')

    def test_float_checkpoint_storage_and_training_precision_are_separate(self):
        with tempfile.TemporaryDirectory(prefix='myptq-dtypes-') as folder, self.fixture.context():
            root = Path(folder)
            base = torch.load(self.fixture.data(root), weights_only=False)
            for precision, dtype in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
                ckpt = root / f'{precision}.ckpt'
                payload = dict(base, hyper_parameters={'trainer': {'precision': finetune.PRECISIONS[precision]}},
                               state_dict={n: t.to(dtype) for n, t in base['state_dict'].items()})
                torch.save(payload, ckpt)
                out = root / f'{precision}-result'
                ptq.main(self.fixture.command(root, ckpt, out) + ['--quant-format', 'W4A32'])
                metadata = json.loads((out / 'myptq_metadata.json').read_text())
                self.assertEqual(metadata['finetune_precision'], precision)
                self.assertEqual(metadata['source_weight_dtypes'], [str(dtype)])
                self.assertEqual(metadata['ptq_compute_dtype'], 'fp32')
                saved = torch.load(out / 'myptq.ckpt', weights_only=False)
                self.assertTrue(all(t.dtype == torch.float32 for t in saved['state_dict'].values()))

    def test_all_requested_formats_calibrate_and_reload_exactly(self):
        cases = [('W8A32', None, 'partial-rotation', '128'), ('W4A32', None, 'full-rotation', 'none'),
                 ('W8A8', 'observer', 'symmetric', '32'), ('W8A8', 'sequential', 'partial-rotation', '64'),
                 ('W4A8', 'observer', 'full-rotation', 'none'), ('W4A8', 'sequential', 'symmetric', '128'),
                 ('W4A4', 'observer', 'partial-rotation', '64'), ('W4A4', 'sequential', 'full-rotation', '32')]
        with tempfile.TemporaryDirectory(prefix='myptq-formats-') as folder, self.fixture.context():
            root = Path(folder)
            ckpt = self.fixture.data(root)
            parent = torch.load(ckpt, weights_only=False)['state_dict']
            for index, (name, calibration, method, group) in enumerate(cases):
                with self.subTest(name=name, calibration=calibration):
                    out = root / f'fresh-{index}'
                    argv = self.fixture.command(root, ckpt, out) + ['--quant-format', name, '--weight-scope', 'all-weights',
                        '--group-size', group, '--quantization', method, '--rotation-transform', 'hadamard', '--weight-init', 'mse']
                    if calibration:
                        argv += ['--calibration-method', calibration, '--activation-percentile', '95', '--calibration-windows', '8']
                    ptq.main(argv)
                    saved = torch.load(out / 'myptq.ckpt', weights_only=False)
                    self.assertNotIn('float_state_dict', saved)
                    reference = self.fixture.model()
                    reference.load_state_dict(parent, strict=True)
                    weights.apply_ptq(reference, ptq.config_from_dict(saved['config']))
                    for n, t in reference.state_dict().items():
                        self.assertTrue(torch.equal(saved['state_dict'][n].cpu(), t.cpu()), n)
                    restored = root / f'loaded-{index}'
                    ptq.main(self.fixture.command(root, out / 'myptq.ckpt', restored) + ['--load-calibrated'])
                    with np.load(out / 'myptq_predictions.npz') as a, np.load(restored / 'myptq_predictions.npz') as b:
                        np.testing.assert_array_equal(a['logits'], b['logits'])
                    metadata = json.loads((out / 'myptq_metadata.json').read_text())
                    self.assertEqual(metadata['quant_format'], name)
                    self.assertEqual(metadata['ptq']['quantized_weight_tensors'], 28)
                    self.assertEqual(metadata['activation_quantizers'], 42 if calibration == 'observer' else 32 if calibration else 0)

    def test_sweep_accepts_all_five_formats(self):
        with tempfile.TemporaryDirectory(prefix='myptq-plan-') as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            plan = sweep.main(['--checkpoint', str(root / 'float.ckpt'), '--data-dir', str(root), '--output-dir', str(root / 'plan'),
                '--quant-formats', *ptq.QUANT_FORMATS, '--group-sizes', 'none', '--quantizations', 'symmetric'])
            self.assertEqual(plan['cases_count'], 5)
            for job in plan['cases']:
                cfg = ptq.configuration(ptq.parser().parse_args(job['argv'][2:]))
                self.assertEqual(cfg.to_dict(), job['config'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
