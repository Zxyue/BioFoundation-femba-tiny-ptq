"""CUDA 入口检查与运行信息记录；不锁死环境补丁版本。"""
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import torch


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'file': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def require_cuda(device):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Run in the remote CUDA environment.')


def load_mamba():
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    return Mamba, selective_scan_fn


def runtime_record():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'mamba_ssm': version('mamba-ssm'), 'brevitas': version('brevitas')}


def implementation_record(repo):
    # 记录代码来源供结果比较；加载 checkpoint 时不要求哈希完全相同。
    base = Path(__file__).resolve().parent
    names = ('ptq.py', 'ptq_quant.py', 'ptq_calibration.py', 'ptq_activation_observer.py',
             'ptq_activation_sequential.py', 'ptq_runtime.py')
    result = {'framework': {n: sha_file(base / n) for n in names}}
    model = Path(repo) / 'models/FEMBA.py'
    result['model'] = sha_file(model) if model.is_file() else None
    result['mamba_sources'] = {}
    for name in ('mamba_ssm.modules.mamba_simple', 'mamba_ssm.ops.selective_scan_interface'):
        path = getattr(sys.modules.get(name), '__file__', None)
        result['mamba_sources'][name] = sha_file(path) if path and Path(path).is_file() else None
    return result
