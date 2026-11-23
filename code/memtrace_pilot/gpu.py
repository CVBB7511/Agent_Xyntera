"""Serialize local inference and charge occupied time, including failed probes."""

import contextlib
import csv
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from common import OUT, ROOT, SRC, context_window, read, write

MEMORY_STOP_PERCENT = 75
ALLOCATOR_TOTAL_PERCENT = 70
MEMORY_QUERY_ATTEMPTS = 3


class GPUUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RecoveryGPU:
    uuid: str
    memory_used: int
    memory_total: int


def below_memory_limit(used, total):
    if total <= 0 or not 0 <= used <= total:
        raise ValueError("invalid GPU memory telemetry")
    return used * 100 < MEMORY_STOP_PERCENT * total


def select_recovery_gpu(query):
    rows = list(
        csv.reader(
            query("--query-gpu=index,uuid,name,memory.used,memory.total").splitlines(),
            skipinitialspace=True,
        )
    )
    if not rows or any(len(row) != 5 for row in rows):
        raise ValueError("incomplete GPU inventory")
    devices = {row[1] for row in rows}
    if len(devices) != len(rows) or not all(uid.startswith("GPU-") for uid in devices):
        raise ValueError("invalid GPU inventory")
    below_memory_threshold = 0
    candidates = []
    for index, uid, name, used_str, total_str in rows:
        used, total = map(int, (used_str, total_str))
        memory_available = below_memory_limit(used, total)
        below_memory_threshold += int(memory_available)
        print(
            f"GPU {index}: {name}, {used}/{total} MiB, below {MEMORY_STOP_PERCENT}% memory={memory_available}",
            flush=True,
        )
        if (
            memory_available
            and "A800" in name.upper()
            and used * 100 < ALLOCATOR_TOTAL_PERCENT * total
        ):
            candidates.append(
                ((used / total, int(index)), RecoveryGPU(uid, used, total))
            )
    if below_memory_threshold < 2:
        raise ValueError(
            f"BYL requires at least two GPUs below {MEMORY_STOP_PERCENT}% memory; found {below_memory_threshold}"
        )
    if not candidates:
        raise ValueError("no A800 with enough allocator headroom")
    return min(candidates, key=lambda item: item[0])[1]


def watch_recovery(process, device, record, *, stop=None):
    """Stop this session if the selected card ceases to have spare capacity."""
    missing_samples = 0
    while process.poll() is None and (stop is None or not stop.is_set()):
        try:
            sample = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2 + missing_samples,
            )
            used, total = map(int, sample.strip().split(","))
            qualifies = below_memory_limit(used, total)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            if process.poll() is not None or (stop is not None and stop.is_set()):
                return
            missing_samples += 1
            record["memory_query_failures"] = record.get("memory_query_failures", 0) + 1
            if missing_samples == MEMORY_QUERY_ATTEMPTS:
                reason = f"GPU memory unavailable after {MEMORY_QUERY_ATTEMPTS} consecutive checks: {error}"
                record.update(
                    resource_stop=reason,
                    resource_stop_kind="memory_telemetry_unavailable",
                )
                raise GPUUnavailable(reason) from error
            print(
                f"GPU memory query returned no valid sample ({missing_samples}/{MEMORY_QUERY_ATTEMPTS}); retrying",
                flush=True,
            )
        else:
            if process.poll() is not None:
                return
            missing_samples = 0
            record["peak_memory_mib"] = max(record.get("peak_memory_mib", 0), used)
            if not qualifies:
                reason = f"GPU memory guard stopped the job at {used}/{total} MiB"
                record.update(resource_stop=reason, resource_stop_kind="memory_limit")
                raise GPUUnavailable(reason)
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            continue


def terminate_owned_group(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextlib.contextmanager
def engine_job(target, max_seconds, *, receipt=None, abort_run=None):
    import fcntl

    sys.path.insert(0, str(ROOT / "code/version0_0"))
    from byl_gpu import query

    with (OUT / "gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = OUT / "gpu-time.json"
        times = read(ledger) if ledger.exists() else []
        budget = 50 * 3600 - sum(t["seconds"] for t in times)
        if budget <= 2:
            raise GPUUnavailable("50 GPU-hour cap reached")
        duration = min(int(budget), max_seconds)
        try:
            initial = select_recovery_gpu(query)
            rechecked = select_recovery_gpu(query)
            if rechecked.uuid != initial.uuid:
                raise ValueError("GPU memory availability changed before launch")
            selected = rechecked.uuid
            fraction = (
                ALLOCATOR_TOTAL_PERCENT / 100
                - rechecked.memory_used / rechecked.memory_total
            )
            if fraction <= 0:
                raise ValueError("insufficient allocation headroom for the model")
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise GPUUnavailable(str(exc)) from exc
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": selected,
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".conda/runtime/inductor"),
            "TRITON_CACHE_DIR": str(ROOT / ".conda/runtime/triton"),
            "PILOT_CUDA_MEMORY_FRACTION": str(fraction),
        }
        start = time.monotonic()
        record = dict(
            id=uuid.uuid4().hex,
            start=time.time(),
            gpu=selected,
            seconds=duration,
            state="reserved",
            context_window=context_window(),
            initial_memory_mib=rechecked.memory_used,
            total_memory_mib=rechecked.memory_total,
            allocator_fraction=fraction,
            memory_stop_percent=MEMORY_STOP_PERCENT,
        )
        times.append(record)
        write(ledger, times)
        with (OUT / ("gpu-" + record["id"] + ".log")).open("w") as log:
            process = subprocess.Popen(
                [
                    "timeout",
                    "-k",
                    "5",
                    str(max(1, duration - 1)),
                    sys.executable,
                    str(SRC / "provider.py"),
                    str(target),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stop = threading.Event()

            def enforce_memory_limit():
                try:
                    watch_recovery(process, selected, record, stop=stop)
                except GPUUnavailable as exc:
                    print(str(exc), flush=True)
                    terminate_owned_group(process)
                    if abort_run is not None:
                        abort_run()

            monitor = threading.Thread(
                target=enforce_memory_limit, name="gpu-memory-guard"
            )
            monitor.start()
            try:
                yield process
            finally:
                stop.set()
                monitor.join()
                terminate_owned_group(process)
                record.update(
                    seconds=time.monotonic() - start,
                    state="finished",
                    exit_code=process.returncode,
                )
                write(ledger, times)
                if receipt is not None:
                    write(receipt, record)
                if record.get("resource_stop"):
                    raise GPUUnavailable(record["resource_stop"])
