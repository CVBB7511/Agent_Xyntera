"""Launch a single GPU pilot on BYL after checking the shared GPU pool."""

import csv
import os
import subprocess
import sys
from pathlib import Path

from run import parser


ROOT = Path("/mnt/sata1/beihang-cuda_backup")
MODELS = ROOT / "models"


def query(*args: str) -> str:
    return subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout


def idle_a800() -> str:
    rows = list(csv.reader(query("--query-gpu=index,uuid,name,memory.used,utilization.gpu").splitlines(),
                           skipinitialspace=True))
    running = {row.strip() for row in query("--query-compute-apps=gpu_uuid").splitlines() if row.strip()}
    if not rows or any(len(row) != 5 for row in rows):
        raise ValueError("Incomplete GPU inventory")
    reported = {row[1] for row in rows}
    if len(reported) != len(rows) or not running <= reported:
        raise ValueError("Could not match compute occupancy to the GPU inventory")
    idle = []
    for index, uuid, name, memory, utilization in rows:
        if not uuid.startswith("GPU-"):
            raise ValueError(f"Unexpected GPU identifier: {uuid}")
        # Allow small driver allocations. Any compute process or utilization rules out a card.
        used, busy = int(memory), int(utilization)
        print(f"GPU {index}: {name}, {used} MiB, {busy}%, compute process={uuid in running}", flush=True)
        if uuid not in running and used <= 128 and busy == 0:
            idle.append((used, int(index), uuid, name))
    if len(idle) < 2:
        raise ValueError(f"BYL requires at least two idle GPUs; found {len(idle)}")
    eligible = [gpu for gpu in idle if "A800" in gpu[3].upper()]
    if not eligible:
        raise ValueError("No idle A800 found")
    return min(eligible)[2]


def within(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path outside {root}: {path}")


def main(argv: list[str]) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command not in {"run", "model-smoke"} or not args.model_dir or args.device != "cuda:0":
        cli.error("Use run or model-smoke with --model-dir and cuda:0")
    try:
        if not ROOT.parent.is_mount():
            raise ValueError("BYL data disk is not mounted")
        within(Path.cwd(), ROOT)
        within(Path(__file__), ROOT)
        within(Path(sys.executable), ROOT)
        within(args.model_dir, MODELS)
        for case in getattr(args, "case", []):
            within(case, ROOT)
        if args.command == "run":
            within(args.output, ROOT)
            within(Path(args.test_python), ROOT)
        local_cache = ROOT / ".conda" / "runtime"
        local_cache.mkdir(parents=True, exist_ok=True)
        gpu = idle_a800()
        if idle_a800() != gpu:
            raise ValueError("GPU occupancy changed during the pre-launch check")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        cli.exit(1, f"BYL launch stopped: {exc}\n")
    print(f"Launching on {gpu}", flush=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "TMPDIR": str(local_cache),
                "HF_HOME": str(local_cache / "huggingface"), "XDG_CACHE_HOME": str(local_cache),
                "TRITON_CACHE_DIR": str(local_cache / "triton"), "PYTHONNOUSERSITE": "1"})
    return subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run.py")), *argv], env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
