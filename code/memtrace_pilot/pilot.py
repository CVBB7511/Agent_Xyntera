"""BYL entry point. All paths are anchored to the project mount."""

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (
    CONTEXT_WINDOWS,
    DATA,
    DEEP,
    LEGACY_CONTEXT_WINDOW,
    OUT,
    PREFIX,
    ROOT,
    SK,
    SRC,
    context_window,
    digest,
    read,
    result_directory,
    write,
)


LEGACY_OVERFLOW_ATTEMPT = "d460ad75f21942f99467bcceb5e674c4"


def scoped_environment():
    if Path.cwd().resolve() != ROOT or not ROOT.parent.is_mount():
        raise RuntimeError("run from the mounted BYL project root")
    os.umask(0o077)
    result_directory(OUT).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HOME=str(ROOT / ".conda/pilot-home"),
        XDG_CACHE_HOME=str(ROOT / ".conda/runtime"),
        TMPDIR=str(ROOT / ".conda/tmp"),
        GIT_LFS_SKIP_SMUDGE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONNOUSERSITE="1",
    )


def manifest():
    import json
    import hashlib

    files = (
        list((ROOT / "vendor/MemTrace/src").rglob("*.py"))
        + list(SRC.glob("*.py"))
        + list(SRC.glob("*.sh"))
    )
    files += list((ROOT / "vendor/MemTrace/configs").rglob("*.json"))
    files.extend(
        [
            ROOT / "models/Qwen3.5-9B/config.json",
            ROOT / "models/Qwen3.5-9B/chat_template.jinja",
            ROOT / "vendor/MemTrace/src/memtrace/_build_receipt.json",
            OUT / "platform.lock",
        ]
    )
    current = OUT / "code-manifest.json"
    snapshots = OUT / "manifests"

    def archive(mapping):
        fingerprint = hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()
        ).hexdigest()
        write(snapshots / (fingerprint + ".json"), mapping)
        return fingerprint

    if current.exists():
        archive(read(current))
    mapping = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    fingerprint = archive(mapping)
    write(current, mapping)
    return fingerprint


def controls(selected):
    if selected in (None, "DeepSWE"):
        import deepswe

        for task in DEEP:
            destination = OUT / "controls" / task / "status.json"
            try:
                deepswe.controls(task, read(OUT / "inputs" / (task + ".json")))
                write(destination, {"ready": True})
            except Exception as exc:
                write(destination, {"ready": False, "error": str(exc)})
    for benchmark, module in (("SWE-EVO", "evo"), ("SWE-Milestone", "mile")):
        if selected in (None, benchmark):
            subprocess.run(
                [PREFIX / "bin/python", "-c", f"import {module}; {module}.controls()"],
                cwd=SRC,
                check=True,
            )


def archive_legacy_overflow():
    """Allow one replacement for the documented pre-fix infrastructure failure."""
    import json

    destination = OUT / "runs" / SK / "memtrace"
    archive = OUT / "attempt-history" / SK / "memtrace" / LEGACY_OVERFLOW_ATTEMPT
    if archive.exists():
        print(f"Legacy context overflow already archived at {archive}", flush=True)
        return
    attempt_file = destination / "attempt.json"
    if not attempt_file.is_file():
        return
    stamp = read(attempt_file)
    if stamp.get("id") != LEGACY_OVERFLOW_ATTEMPT:
        return
    if (
        "seconds" not in stamp
        or (destination / "grade.json").exists()
        or any((destination / "nodes").glob("*/snapshot.json"))
        or any((destination / "nodes").glob("*/grade.json"))
    ):
        raise RuntimeError("legacy run is unfinished or has official submissions")
    receipt = destination / "trace/memtrace/result.json"
    if receipt.exists():
        result = read(receipt)
    else:
        with (destination / "agent.log").open(encoding="utf-8") as log:
            results = [
                json.loads(line) for line in log if line.startswith('{"build_identity"')
            ]
        result = results[-1] if results else {}
    metrics = result.get("metrics", {})
    if (
        result.get("completion_verdict") != "INCOMPLETE"
        or metrics.get("provider_turn_failure_retry_count") != 5
        or metrics.get("provider_turn_failures_exhausted_count") != 1
    ):
        raise RuntimeError("cannot verify the known pre-fix context failure")
    archive.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(archive)
    print(f"Archived legacy context-overflow attempt at {archive}", flush=True)


def archive_monitor_interruption(results, task, arm):
    import re

    destination = results / "runs" / task / arm
    attempt_file = destination / "attempt.json"
    gpu_file = destination / "gpu.json"
    if not attempt_file.is_file() or not gpu_file.is_file():
        return
    gpu = read(gpu_file)
    failure = gpu.get("resource_stop", "")
    if gpu.get("state") != "finished" or not (
        gpu.get("resource_stop_kind") == "memory_telemetry_unavailable"
        or ("nvidia-smi" in failure and "timed out" in failure)
    ):
        return
    if (
        (destination / "grade.json").exists()
        or (destination / "model.patch").exists()
        or any((destination / "nodes").glob("*/snapshot.json"))
        or any((destination / "nodes").glob("*/grade.json"))
    ):
        raise RuntimeError(f"{task} {arm} already has a submission or score")
    attempt_id = read(attempt_file).get("id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", attempt_id):
        raise RuntimeError("cannot verify the interrupted attempt ID")
    archive = results / "attempt-history" / task / arm / attempt_id
    if archive.exists():
        raise RuntimeError(f"attempt archive already exists at {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(archive)
    print(f"Archived GPU telemetry interruption at {archive}", flush=True)


def verify_memtrace_install():
    receipt = read(ROOT / "vendor/MemTrace/src/memtrace/_build_receipt.json")
    installed = subprocess.check_output(
        [
            PREFIX / "bin/python",
            "-c",
            "from memtrace.build_identity import current_build_identity; "
            "print(current_build_identity(require_receipt=True).build_id)",
        ],
        text=True,
    ).strip()
    if installed != receipt["build_id"]:
        raise RuntimeError(
            "synced MemTrace build differs from installed package; run refresh-memtrace.sh"
        )


def run_or_grade(
    action, selected=None, selected_task=None, time_limit=None, *, retry_monitor=False
):
    import deepswe
    from reporting import matrix, report
    from trial import attempt
    from gpu import GPUUnavailable

    results = result_directory(OUT)
    tasks = [
        (bench, task)
        for bench, task in matrix()
        if (selected is None or selected == bench)
        and (selected_task is None or selected_task == task)
    ]
    if selected_task is not None and not tasks:
        raise ValueError(f"unknown task in the selected benchmark: {selected_task}")
    for bench, task in tasks:
        status = OUT / "controls" / task / "status.json"
        if not status.exists() or not read(status).get("ready"):
            if selected_task is not None:
                raise RuntimeError(f"{bench} {task}: controls unavailable or failed")
            print(f"{bench} {task}: controls unavailable or failed", flush=True)
            continue
        for arm in ("native", "memtrace"):
            destination = results / "runs" / task / arm
            if retry_monitor:
                archive_monitor_interruption(results, task, arm)
            destination.mkdir(parents=True, exist_ok=True)
            for basename in ("launch-blocked", action + "-error"):
                stale = destination / (basename + ".json")
                if stale.exists():
                    stale.rename(destination / (f"{basename}-{time.time_ns()}.json"))
            try:
                print(f"{action}: {bench} {task} {arm}", flush=True)
                if action == "run":
                    for name in ("smoke-context.json", "smoke-tools.json"):
                        if not (results / name).is_file():
                            raise RuntimeError(
                                f"missing {context_window()}-token provider probe {name}"
                            )
                    if bench == "SWE-Milestone":
                        import mile

                        mile.run(arm, time_limit=time_limit)
                    else:
                        info = read(OUT / "inputs" / (task + ".json"))
                        if time_limit is not None:
                            info["timeout"] = min(info["timeout"], time_limit)
                        prompt = (
                            (
                                ROOT / "deep-swe/tasks" / task / "instruction.md"
                            ).read_text()
                            if task in DEEP
                            else read(DATA / "evo" / task / "row.json")[
                                "problem_statement"
                            ]
                        )
                        if task in DEEP:
                            prompt += "\nCommit your final changes. The evaluator collects git diff --binary from the base commit to HEAD."
                        attempt(
                            task,
                            arm,
                            info,
                            prompt,
                            "/app" if task in DEEP else "/testbed",
                        )
                patch = destination / "model.patch"
                attempt_file = destination / "attempt.json"
                if attempt_file.exists() and read(attempt_file).get(
                    "infrastructure_issue"
                ):
                    continue
                if (destination / "grade.json").exists():
                    continue
                if bench == "SWE-Milestone":
                    subprocess.run(
                        [
                            PREFIX / "bin/python",
                            "-c",
                            f"import mile; mile.grade_saved({arm!r})",
                        ],
                        cwd=SRC,
                        check=True,
                    )
                elif patch.exists():
                    if task in DEEP:
                        deepswe.grade(
                            task,
                            patch,
                            destination,
                            read(OUT / "inputs" / (task + ".json")),
                        )
                    else:
                        subprocess.run(
                            [
                                PREFIX / "bin/python",
                                str(SRC / "evo_score.py"),
                                task,
                                arm,
                            ],
                            check=True,
                        )
            except GPUUnavailable as exc:
                filename = (
                    "run-error.json"
                    if (destination / "gpu.json").is_file()
                    else "launch-blocked.json"
                )
                write(destination / filename, {"error": str(exc)})
                report()
                raise
            except Exception as exc:
                print(f"{action} error: {bench} {task} {arm}: {exc}", flush=True)
                write(
                    destination / (action + "-error.json"),
                    {"error": str(exc), "traceback": traceback.format_exc()},
                )
            report()
        if bench == "SWE-EVO" and all(
            (results / "runs" / task / arm / "grade.json").exists()
            for arm in ("native", "memtrace")
        ):
            from evo import release_image

            release_image(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "preflight",
            "smoke",
            "recovery-check",
            "run",
            "grade",
            "report",
        ],
    )
    parser.add_argument("--private", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", choices=["DeepSWE", "SWE-EVO", "SWE-Milestone"])
    parser.add_argument("--context-window", type=int, choices=CONTEXT_WINDOWS)
    parser.add_argument(
        "--task", help="select exactly one frozen task for run or grade"
    )
    parser.add_argument(
        "--time-limit", type=int, help="maximum seconds per arm for run"
    )
    parser.add_argument(
        "--smoke-input-tokens", type=int, help="target input length for the smoke probe"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="also make one model call in recovery-check",
    )
    parser.add_argument(
        "--retry-context-overflow",
        action="store_true",
        help="archive the documented pre-fix Milestone failure and replace it once",
    )
    parser.add_argument(
        "--retry-monitor-interruption",
        action="store_true",
        help="retry a single task after archiving an unsubmitted GPU telemetry interruption",
    )
    args = parser.parse_args()
    if args.context_window is not None:
        os.environ["PILOT_CONTEXT_WINDOW"] = str(args.context_window)
    if args.task and args.command not in {"run", "grade"}:
        parser.error("--task requires run or grade")
    if args.retry_monitor_interruption and (args.command != "run" or not args.task):
        parser.error("--retry-monitor-interruption requires run --task")
    if args.time_limit is not None and (args.command != "run" or args.time_limit <= 0):
        parser.error("--time-limit requires run and a positive number of seconds")
    if args.smoke_input_tokens is not None:
        if (
            args.command != "smoke"
            or not 1024 <= args.smoke_input_tokens <= context_window() - 3072
        ):
            parser.error(
                "--smoke-input-tokens requires smoke and room for its output reserve"
            )
        os.environ["PILOT_SMOKE_INPUT_TOKENS"] = str(args.smoke_input_tokens)
    if args.retry_context_overflow and (args.command, args.benchmark) != (
        "run",
        "SWE-Milestone",
    ):
        parser.error("--retry-context-overflow requires run --benchmark SWE-Milestone")
    if args.retry_context_overflow and context_window() != LEGACY_CONTEXT_WINDOW:
        parser.error(
            "--retry-context-overflow applies only to the 131072-token results"
        )
    if args.generate and args.command != "recovery-check":
        parser.error("--generate requires recovery-check")
    scoped_environment()
    action = args.command
    if action in ("preflight", "run", "grade") and not args.private:
        tail = ["--benchmark", args.benchmark] if args.benchmark else []
        if args.retry_context_overflow:
            tail.append("--retry-context-overflow")
        if args.retry_monitor_interruption:
            tail.append("--retry-monitor-interruption")
        if args.task:
            tail.extend(["--task", args.task])
        if args.time_limit is not None:
            tail.extend(["--time-limit", str(args.time_limit)])
        subprocess.run(
            [
                "bash",
                SRC / "project-docker.sh",
                sys.executable,
                __file__,
                action,
                "--private",
                *tail,
            ],
            check=True,
        )
        return
    if action == "prepare":
        subprocess.run([PREFIX / "bin/python", SRC / "benchdata.py"], check=True)
        subprocess.run(["bash", SRC / "evaluator-env.sh"], check=True)
        manifest()
    elif action == "preflight":
        controls(args.benchmark)
    elif action == "smoke":
        from gpu import engine_job

        with engine_job("smoke", 1800) as p:
            if p.wait():
                raise RuntimeError("GPU probe failed; see gpu-time.json and its log")
    elif action == "recovery-check":
        verify_memtrace_install()
        subprocess.run([sys.executable, str(SRC / "recovery_check.py")], check=True)
        if args.generate:
            from gpu import engine_job

            with engine_job("recovery-check", 180) as p:
                if p.wait():
                    raise RuntimeError(
                        "recovery generation failed; see gpu-time.json and its log"
                    )
    elif action == "report":
        from reporting import report

        report()
    else:
        import fcntl

        if action == "run":
            verify_memtrace_install()
        manifest()
        lock_name = (
            "evo-regrade.lock"
            if args.benchmark == "SWE-Milestone"
            else "workspace-probe.lock"
        )
        with (OUT / lock_name).open("a") as phase_lock:
            fcntl.flock(phase_lock, fcntl.LOCK_EX)
            if args.retry_context_overflow:
                archive_legacy_overflow()
            run_or_grade(
                action,
                args.benchmark,
                args.task,
                args.time_limit,
                retry_monitor=args.retry_monitor_interruption,
            )


if __name__ == "__main__":
    main()
