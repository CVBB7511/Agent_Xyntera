"""Tiny deterministic PyTorch test; CUDA execution requires a Slurm allocation."""
import argparse
import json
import os
import socket
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA testing must run inside a Slurm allocation")

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(42)
    started = time.monotonic()
    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to fall back to CPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one Slurm-visible GPU")
    x = torch.randn(64, 4, device=device)
    target = x @ torch.tensor([[1.0], [2.0], [-1.0], [0.5]], device=device)
    model = torch.nn.Linear(4, 1, bias=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_loss = torch.nn.functional.mse_loss(model(x), target).item()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    final_loss = torch.nn.functional.mse_loss(model(x), target).item()
    if not (final_loss < initial_loss and final_loss < 0.01):
        raise RuntimeError(f"Training check failed: {initial_loss} -> {final_loss}")
    result = {
        "status": "PASS", "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(device), "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "steps": 50, "initial_loss": initial_loss, "final_loss": final_loss,
        "elapsed_seconds": time.monotonic() - started,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        # Parent must exist; never overwrite previous experimental results.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
