"""Small allocated-GPU correctness test, with no model or dataset downloads."""
import argparse
import json
import os
import socket
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('A Slurm GPU allocation is required')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback'
    assert torch.cuda.device_count() == 1, 'Expected exactly one visible GPU'
    torch.manual_seed(42)
    started = time.monotonic()
    a, b = torch.randn(256, 256), torch.randn(256, 256)
    ref = a @ b
    actual = a.cuda() @ b.cuda()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), ref, rtol=1e-4, atol=1e-4)
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_array_equal(torch.from_numpy(array).cuda().cpu().numpy(), array)
    model = torch.nn.Conv2d(3, 4, 3).cuda()
    x = torch.randn(2, 3, 32, 32, device='cuda', requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all().item()
    assert all(v.grad is not None and torch.isfinite(v.grad).all().item() for v in model.parameters())
    torch.cuda.synchronize()
    prop = torch.cuda.get_device_properties(0)
    result = {'status': 'PASS', 'job_id': os.environ['SLURM_JOB_ID'],
              'hostname': socket.gethostname(), 'torch': torch.__version__,
              'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': prop.name, 'gpu_memory_bytes': prop.total_memory,
              'compute_capability': list(torch.cuda.get_device_capability(0)),
              'cudnn_version': torch.backends.cudnn.version(),
              'checks': ['cuda_available', 'single_visible_gpu', 'matmul_matches_cpu',
                         'numpy_roundtrip', 'conv2d_backward_finite'],
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'elapsed_seconds': time.monotonic() - started}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
