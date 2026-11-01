"""激活校准：统计后统一冻结，或按前向顺序逐阶段冻结。"""
import numpy as np
import torch


class SequentialController:
    def __init__(self, model, config):
        import ptq_activation_sequential as source
        self.model, self.config, self.source = model, config, source
        self.aq = source.install_act_quant(model, config.activation_bits)
        self.names = {name: name for name in self.aq.q}
        self.set_enabled(False)

    def set_enabled(self, enabled):
        self.aq.force_custom = not enabled
        for q in self.aq.q.values():
            q.calibrating = False
            q.enabled.fill_(enabled and bool(q.initialized))
        self.model.eval()

    def frozen_scales(self):
        if self.aq.force_custom or any(q.calibrating or not bool(q.enabled) or not bool(q.initialized) for q in self.aq.q.values()):
            raise RuntimeError('Sequential calibration is incomplete.')
        result = {n: q.log_range.detach().exp().cpu().clone() / q.levels for n, q in self.aq.q.items()}
        if any(not bool(torch.isfinite(v).all()) or not bool((v > 0).all()) for v in result.values()):
            raise RuntimeError('Invalid activation scale.')
        return result


def install_activations(model, config, *, mamba_type=None, scan_fn=None):
    if config.activation_bits == 32:
        return None
    if config.calibration_method == 'sequential':
        return SequentialController(model, config)
    import ptq_activation_observer as observer
    return observer.install_quantizers(model, config, mamba_type=mamba_type, scan_fn=scan_fn)


def choose_indices(total, config):
    """仅从训练集无放回抽样；两种校准沿用各自的随机数与排序规则。"""
    if total <= 0:
        raise ValueError('Training split is empty.')
    count = min(total, config.calibration_windows)
    if config.calibration_method == 'sequential':
        return sorted(np.random.default_rng(config.calibration_seed).choice(total, count, replace=False).tolist())
    generator = torch.Generator().manual_seed(config.calibration_seed)
    return torch.randperm(total, generator=generator)[:count].tolist()


def calibrate(model, controller, config, batch_factory):
    if config.calibration_method == 'observer':
        import ptq_activation_observer as observer
        return observer.calibrate(model, controller, batch_factory())
    # 多阶段校准重复使用同一批窗口；缓存归一化后的 CPU 数据，控制显存占用。
    cached = [x.detach().cpu() for x in batch_factory()]
    device = next(model.parameters()).device

    def run():
        for x in cached:
            model(x.to(device), torch.zeros_like(x, device=device, dtype=torch.bool))

    controller.aq.force_custom = False
    with torch.no_grad():
        count = controller.source.calibrate_sequential(controller.aq, run, config.activation_bits, config.activation_percentile)
    model.eval()
    if count != len(controller.names):
        raise RuntimeError('Calibration missed activation points.')
    controller.frozen_scales()
    stages = len(controller.aq.stages())
    return {'num_batches': len(cached), 'forward_batches': len(cached) * stages, 'stages': stages,
            'activation_quantizers': count, 'points': controller.names, 'ranges': controller.aq.ranges(),
            'lower_bounds': {n: q.lower for n, q in controller.aq.q.items()}}
