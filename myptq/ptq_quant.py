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

"""Weight-only fake PTQ: grouping, partial/full orthogonal rotation, original modules kept."""
from dataclasses import asdict, dataclass
import hashlib
from fnmatch import fnmatchcase
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PTQConfig:
    """权重与激活设置；分组大小和旋转块大小是两个独立参数。"""
    bits: int = 4
    group_size: int | None = 64
    granularity: str = 'channel'
    method: str = 'symmetric'
    rotation_transform: str = 'qr'
    rotation_size: int = 64
    rotation_seed: int = 2025
    clip_percentile: float = 100.0
    scope: str = 'encoder-linear'
    classifier_bits: int | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    weight_init: str = 'max'
    activation_bits: int = 32
    calibration_method: str = 'observer'
    activation_percentile: float = 99.999
    calibration_windows: int = 512
    calibration_batch_size: int = 32
    calibration_seed: int = 42
    calibration_weight_state: str = 'float'

    def validate(self):
        if self.weight_init not in ('max','mse'):raise ValueError('weight-init must be max or mse.')
        if self.weight_init=='mse' and self.clip_percentile!=100:raise ValueError('MSE searches its own clipping range; use clip-percentile=100.')
        if self.activation_bits not in (4,8,32):raise ValueError('activation-bits must be 4,8,32.')
        if self.calibration_method not in ('observer','sequential'):raise ValueError('Unknown activation calibration method.')
        if self.calibration_weight_state not in ('float','quantized'):raise ValueError('calibration-weight-state must be float or quantized.')
        if not math.isfinite(self.activation_percentile) or not 0<self.activation_percentile<=100:raise ValueError('Invalid activation percentile.')
        if self.activation_bits!=32 and self.calibration_method=='sequential' and self.activation_percentile not in (95.,99.,99.9,99.99,99.999,100.):
            raise ValueError('Sequential calibration supports percentiles 95,99,99.9,99.99,99.999,100.')
        if self.calibration_windows<=0 or self.calibration_batch_size<=0 or self.calibration_seed<0:raise ValueError('Invalid calibration windows/batch/seed.')
        if type(self.bits) is not int or self.bits not in (0, 4, 8):
            raise ValueError('bits must be 0 (float diagnostic), 4 or 8.')
        if self.group_size is not None and (type(self.group_size) is not int or self.group_size not in (32,64,128)):
            raise ValueError('group_size must be none, 32, 64 or 128.')
        if self.granularity not in ('channel', 'tensor'):
            raise ValueError('granularity must be channel or tensor.')
        if self.granularity == 'tensor' and self.group_size is not None:
            raise ValueError('Whole-tensor quantization requires --group-size none.')
        if self.method not in ('symmetric', 'block', 'full'):
            raise ValueError('method must be symmetric, block or full.')
        if self.rotation_transform not in ('qr', 'hadamard'):
            raise ValueError('rotation_transform must be qr or hadamard.')
        if type(self.rotation_size) is not int or self.rotation_size not in (32, 64, 128):
            raise ValueError('Block rotation size must be 32, 64 or 128.')
        if not isinstance(self.rotation_seed, int) or isinstance(self.rotation_seed, bool) or not 0 <= self.rotation_seed < 2**63:
            raise ValueError('rotation_seed must be an integer in [0, 2**63).')
        if not math.isfinite(self.clip_percentile) or not 0 < self.clip_percentile <= 100:
            raise ValueError('clip_percentile must be in (0, 100].')
        if self.scope not in ('encoder-linear', 'all-linear', 'all-weights'):
            raise ValueError('Unknown quantization scope.')
        if self.classifier_bits not in (None, 0, 4, 8):
            raise ValueError('classifier_bits must be unset, 0, 4 or 8.')
        if self.scope == 'encoder-linear' and self.classifier_bits not in (None, 0):
            raise ValueError('To quantize the classifier, choose all-linear or all-weights scope.')
        if self.bits == 0 and self.classifier_bits not in (None, 0):
            raise ValueError('Float diagnostics must not quantize the classifier.')
        return self

    def to_dict(self):
        value = asdict(self)
        value["include"], value["exclude"] = list(self.include), list(self.exclude)
        if self.method != "block":
            value["rotation_size"] = None
        if self.method == "symmetric":
            value["rotation_transform"], value["rotation_seed"] = None, None
        if self.activation_bits==32:
            for field in ('calibration_method','activation_percentile','calibration_windows',
                          'calibration_batch_size','calibration_seed','calibration_weight_state'):
                value[field]=None
        return value


def layer_seed(seed, name):
    # Stable across Python processes and independent of group size / quantization bits.
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    return (seed + offset) % (2**63)


def fwht(x):
    """Normalized Walsh-Hadamard transform on the last dimension, no dense matrix."""
    width = x.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError('Hadamard width must be a positive power of two.')
    shape = x.shape
    value = x
    step = 1
    while step < width:
        pairs = value.reshape(*shape[:-1], width // (2 * step), 2, step)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        value = torch.stack((left + right, left - right), dim=-2).reshape(shape)
        step *= 2
    return value / math.sqrt(width)


class Rotation:
    """离线旋转；Hadamard 补齐后保留全部坐标，逆旋转结束才裁掉补齐列。"""
    @torch.no_grad()
    def __init__(self, width, method='block', transform='qr', block_size=64, seed=2025):
        if not isinstance(width, int) or width <= 0:
            raise ValueError('width must be positive.')
        if method not in ('symmetric', 'block', 'full') or transform not in ('qr', 'hadamard'):
            raise ValueError('Invalid rotation method/transform.')
        self.width, self.method, self.transform = width, method, transform
        if method == 'symmetric':
            self.block_size, self.count = width, 1
        elif method == 'block':
            if block_size not in (32, 64, 128):
                raise ValueError('block_size must be 32, 64 or 128.')
            self.block_size = block_size
            self.count = math.ceil(width / block_size)
        else:
            self.block_size = (1 << (width - 1).bit_length()) if transform == 'hadamard' else width
            self.count = 1
        self.padded_width = self.block_size * self.count
        self.active_width = self.padded_width if method != 'symmetric' and transform == 'hadamard' else width
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.matrix, self.signs = None, None
        if method != 'symmetric' and transform == 'qr':
            matrix = torch.eye(self.block_size, dtype=torch.float64).repeat(self.count, 1, 1)
            for index in range(self.count):
                actual = min(self.block_size, width - index * self.block_size)
                if actual > 1:
                    q, r = torch.linalg.qr(torch.randn(actual, actual, generator=generator, dtype=torch.float64))
                    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
                    matrix[index, :actual, :actual] = q * signs.unsqueeze(0)
            self.matrix = matrix
        elif method != 'symmetric':
            self.signs = torch.randint(0, 2, (self.count, self.block_size), generator=generator).double() * 2 - 1

    def _apply(self, rows, inverse):
        expected = self.active_width if inverse else self.width
        if rows.ndim != 2 or rows.shape[1] != expected:
            raise ValueError(f'Expected matrix width {expected}, got {tuple(rows.shape)}.')
        if self.method == 'symmetric':
            return rows.clone()
        values = F.pad(rows, (0, self.padded_width - rows.shape[1])).reshape(rows.shape[0], self.count, self.block_size)
        if self.transform == 'qr':
            matrix = self.matrix.to(values.device, values.dtype)
            if inverse:
                matrix = matrix.transpose(-1, -2)
            values = torch.einsum('ogb,gbk->ogk', values, matrix)
        else:
            signs = self.signs.to(values.device, values.dtype)
            values = fwht(values) * signs if inverse else fwht(values * signs)
        output_width = self.width if inverse else self.active_width
        return values.reshape(rows.shape[0], self.padded_width)[:, :output_width]

    def forward(self, rows):
        return self._apply(rows, False)

    def inverse(self, rows):
        return self._apply(rows, True)

    def describe(self):
        return {'method': self.method, 'transform': self.transform if self.method != 'symmetric' else None,
                'input_width': self.width, 'rotated_width': self.active_width,
                'padding_width': self.padded_width - self.width,
                'block_size': self.block_size if self.method == 'block' else None,
                'full_rotation_dimension': self.block_size if self.method == 'full' else None,
                'computation_dtype': 'float64'}


@torch.no_grad()
def quantize_rows(rows, bits=4, group_size=64, granularity='channel', percentile=100.0, initialization='max'):
    if rows.ndim != 2 or not rows.is_floating_point() or rows.numel() == 0:
        raise ValueError('Expected a nonempty floating-point 2-D matrix.')
    if not bool(torch.isfinite(rows).all()):
        raise ValueError('Non-finite weights.')
    PTQConfig(bits=bits, group_size=group_size, granularity=granularity, clip_percentile=percentile,weight_init=initialization).validate()
    if bits == 0:
        return rows.clone(), {'bits': 0, 'scales': 0, 'relative_error': 0.0}
    original_shape = rows.shape
    work = rows.detach().double()
    if granularity == 'tensor':
        work = work.reshape(1, -1)
    n_rows, width = work.shape
    size = width if group_size is None else group_size
    count = math.ceil(width / size)
    # 每个输出通道沿输入列分组；尾组的补零不参与百分位统计。
    grouped = F.pad(work, (0, count * size - width)).reshape(n_rows, count, size)
    if percentile < 100:
        clip = torch.quantile(grouped.abs(), percentile / 100, dim=-1, keepdim=True)
        tail = width % size
        if tail:
            clip[:, -1] = torch.quantile(work[:, -tail:].abs(), percentile / 100, dim=-1, keepdim=True)
        grouped = grouped.clamp(min=-clip, max=clip)
    qmax = 2 ** (bits - 1) - 1
    scale = (grouped.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    if initialization == 'mse':
        # 各组独立搜索截断范围，按重建误差选择尺度；没有反向传播。
        base = scale.clone()
        best_error = torch.full_like(scale, float('inf'))
        best_scale = scale.clone()
        for ratio in torch.linspace(0.3,1.0,36).tolist():
            candidate = (base * ratio).clamp_min(1e-12)
            trial = (grouped / candidate).round().clamp(-qmax, qmax) * candidate
            error = (trial - grouped).square().sum(-1, keepdim=True)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, candidate, best_scale)
        scale = best_scale
    codes = (grouped / scale).round().clamp(-qmax, qmax)
    restored = (codes * scale).reshape(n_rows, count * size)[:, :width].reshape(original_shape)
    error = restored - rows.double()
    stats = {'bits': bits, 'integer_range': [-qmax, qmax], 'scales': n_rows * count,
             'group_size': group_size, 'granularity': granularity, 'clip_percentile': percentile,'weight_init':initialization,
             'scale_min': float(scale.min()), 'scale_max': float(scale.max()),
             'mse': float(error.square().mean()),
             'relative_error': float(error.norm() / rows.double().norm().clamp_min(1e-30))}
    return restored.to(rows.dtype), stats


@torch.no_grad()
def effective_weight(weight, config, seed, rotate=True):
    config.validate()
    if weight.ndim < 2 or not weight.is_floating_point() or weight.numel() == 0 or not bool(torch.isfinite(weight).all()):
        raise ValueError('Expected finite floating-point weights with at least two dimensions.')
    rows = weight.detach().double().reshape(weight.shape[0], -1)
    rotation = Rotation(rows.shape[1], config.method if rotate else 'symmetric',
                        config.rotation_transform, config.rotation_size, seed)
    rotated = rotation.forward(rows)
    quantized, stats = quantize_rows(rotated, config.bits, config.group_size,
                                     config.granularity, config.clip_percentile,config.weight_init)
    result = rotation.inverse(quantized).reshape(weight.shape).to(weight.dtype)
    delta = result.double() - weight.double()
    stats.update(rotation=rotation.describe(), seed=seed,
                 effective_relative_error=float(delta.norm() / weight.double().norm().clamp_min(1e-30)),
                 effective_mse=float(delta.square().mean()),
                 representation='original-basis floating-point Q(W R) R^T; fake quant; no integer kernel')
    return result, stats


@torch.no_grad()
def apply_ptq(model, config):
    """Modify selected weights once; leave original modules/Parameters/state keys intact."""
    from dataclasses import replace
    config.validate()
    if getattr(model, '_myptq_applied', False):
        raise RuntimeError('PTQ already applied; reload the original floating-point checkpoint for every case.')
    selected, skipped = [], []
    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv = isinstance(module, (nn.Conv1d, nn.Conv2d))
        if not (is_linear or is_conv):
            continue
        in_scope = (is_linear and name.startswith('mamba_blocks.')) if config.scope == 'encoder-linear' else (is_linear or (is_conv and config.scope == 'all-weights'))
        if not in_scope:
            skipped.append({'name': name, 'reason': 'outside_scope'})
            continue
        if config.include and not any(fnmatchcase(name, pattern) for pattern in config.include):
            skipped.append({'name': name, 'reason': 'not_in_include_patterns'})
            continue
        if any(fnmatchcase(name, pattern) for pattern in config.exclude):
            skipped.append({'name': name, 'reason': 'excluded_by_pattern'})
            continue
        bits = config.bits
        if name.startswith('classifier.') or name == 'classifier':
            bits = config.classifier_bits if config.classifier_bits is not None else bits
            if bits == 0 and config.bits != 0:
                skipped.append({'name': name, 'reason': 'classifier_float'})
                continue
        if nn.utils.parametrize.is_parametrized(module, 'weight'):
            raise ValueError(f'{name} is parametrized; start from a floating-point checkpoint.')
        selected.append((name, module, bits))
    if not selected:
        raise ValueError('No weights selected by scope.')
    reports = []
    for name, module, bits in selected:
        local = replace(config, bits=bits, classifier_bits=None)
        seed = layer_seed(config.rotation_seed, name)
        # 旋转只用于 Linear；卷积在三种方法中均使用普通对称量化。
        new_weight, report = effective_weight(module.weight, local, seed, rotate=isinstance(module, nn.Linear))
        module.weight.copy_(new_weight)
        report.update(name=name, module_type=type(module).__name__, weight_shape=list(module.weight.shape))
        reports.append(report)
    if config.bits != 0:
        model._myptq_applied = True
    return {'config': config.to_dict(), 'layers': reports, 'skipped': skipped,
            'quantized_weight_tensors': sum(r['bits'] != 0 for r in reports),
            'rotation_only_on_linear': True, 'activation_quantized': False,
            'original_module_and_parameter_objects_preserved': True}
