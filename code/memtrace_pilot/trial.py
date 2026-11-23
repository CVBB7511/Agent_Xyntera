import fcntl
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from common import (
    DEEP,
    OUT,
    PREFIX,
    ROOT,
    SK,
    SRC,
    cmd,
    context_window,
    digest,
    private_pull,
    read,
    result_directory,
    write,
)
from gpu import engine_job


def observe_milestones(destination, container, image, base, done):
    pending = set(read(OUT / "inputs/milestone.json")["order"])
    while pending:
        last = done.is_set()
        for mid in sorted(pending):
            tag = "refs/tags/agent-impl-" + mid
            if subprocess.run(
                ["docker", "exec", container, "git", "rev-parse", "--verify", tag],
                capture_output=True,
            ).returncode:
                continue
            folder = destination / "nodes" / mid
            try:
                cmd(
                    [
                        PREFIX / "bin/python",
                        SRC / "mile.py",
                        "capture",
                        mid,
                        folder,
                        container,
                        image,
                        base,
                    ]
                )
            except (OSError, subprocess.SubprocessError) as exc:
                write(folder / "capture-error.json", {"error": str(exc)})
            pending.remove(mid)
        if last:
            break
        done.wait(2)


def attempt(task, arm, info, prompt, cwd, *, probe=False, prepare_only=False):
    base = info["base"]
    if prepare_only:
        destination = OUT / "workspace-probes" / task
    else:
        destination = (
            result_directory(OUT) / ("harness-probes" if probe else "runs") / task / arm
        )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "attempt.json").exists():
            return
        image = private_pull(info["image"])
        with tempfile.TemporaryDirectory(
            dir=ROOT / ".conda/tmp", prefix="model-"
        ) as ipc:
            sock = Path(ipc) / "service.sock"
            sandbox = "pilot-agent-" + uuid.uuid4().hex
            work = destination / "trace"
            work.mkdir(exist_ok=True)
            statement = destination / "prompt.txt"
            statement.write_text(prompt)
            profile = (
                ROOT
                / "vendor/MemTrace/configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
            )
            args = [
                "docker",
                "create",
                "--network",
                "none",
                "--name",
                sandbox,
                "-w",
                cwd,
                "--env",
                "OMP_NUM_THREADS=2",
                "--env",
                "LITELLM_LOCAL_MODEL_COST_MAP=True",
                "--env",
                "MSWEA_GLOBAL_CONFIG_DIR=/tmp/mini",
                "-v",
                f"{PREFIX}:{PREFIX}:ro",
                "-v",
                f"{SRC / 'worker.py'}:/pilot/worker.py:ro",
                "-v",
                f"{profile}:/pilot/profile.json:ro",
                "-v",
                f"{statement}:/input/task.txt:ro",
                "-v",
                f"{ipc}:/model:ro",
                "-v",
                f"{work}:/run-out",
                image,
                "sleep",
                "infinity",
            ]
            cmd(args)
            try:
                cmd(["docker", "start", sandbox])
                if probe:
                    cmd(
                        [
                            "docker",
                            "exec",
                            sandbox,
                            "bash",
                            "-ec",
                            "mkdir -p /tmp/pilot-toy; cd /tmp/pilot-toy; git init -q; echo 18 > input.txt; git add .; git -c user.name=pilot -c user.email=pilot@localhost commit -qm initial",
                        ]
                    )
                    cwd = "/tmp/pilot-toy"
                    base = cmd(
                        [
                            "docker",
                            "exec",
                            "-w",
                            cwd,
                            sandbox,
                            "git",
                            "rev-parse",
                            "HEAD",
                        ]
                    )
                elif task == SK:
                    mid = read(OUT / "inputs/milestone.json")["order"][0]
                    cmd(
                        [
                            PREFIX / "bin/python",
                            SRC / "mile.py",
                            "truncate",
                            mid,
                            destination / "masking",
                            sandbox,
                            image,
                            base,
                        ]
                    )
                elif cwd == "/testbed":
                    # Benchmark image versions may start at their environment setup commit.
                    # Materialize only the frozen base and discard refs to later solutions.
                    script = """set -eu
git config --global --add safe.directory /testbed
git checkout -q --detach "$1"
git clean -fdq
git remote | while read -r name; do git remote remove "$name"; done
git for-each-ref --format="delete %(refname)" | git update-ref --no-deref --stdin
git reflog expire --expire=now --all
git gc --prune=now --quiet"""
                    cmd(
                        ["docker", "exec", sandbox, "bash", "-c", script, "pilot", base]
                    )
                if (
                    cmd(
                        [
                            "docker",
                            "exec",
                            "-w",
                            cwd,
                            sandbox,
                            "git",
                            "rev-parse",
                            "HEAD",
                        ]
                    )
                    != base
                ):
                    raise ValueError("agent image HEAD differs from frozen base")
                python = str(PREFIX / "bin/python")
                agent_exec = ["docker", "exec", sandbox]
                if cwd == "/testbed" and task != SK:
                    # The upstream SWE-EVO verifier activates this environment
                    # before invoking tests. Give both agents the same toolchain.
                    agent_exec.extend(
                        [
                            "bash",
                            "-ec",
                            'source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed; exec "$@"',
                            "pilot",
                        ]
                    )
                cmd(
                    [
                        *agent_exec,
                        python,
                        "-c",
                        "import litellm,memtrace,minisweagent,httpx,yaml",
                    ]
                )
                if prepare_only:
                    write(destination / "ready.json", {"base": base, "image": image})
                    return

                def stop_sandbox():
                    subprocess.run(
                        ["docker", "stop", "-t", "1", sandbox],
                        capture_output=True,
                        timeout=10,
                    )

                with engine_job(
                    sock,
                    info["timeout"] + 160,
                    receipt=destination / "gpu.json",
                    abort_run=stop_sandbox,
                ) as process:
                    for _ in range(150):
                        if sock.exists():
                            break
                        if process.poll() is not None:
                            raise RuntimeError(
                                "inference process exited before first request"
                            )
                        time.sleep(1)
                    if not sock.exists():
                        raise RuntimeError("provider socket startup timed out")
                    log = result_directory(OUT) / "provider.jsonl"
                    offset = log.stat().st_size if log.exists() else 0
                    command = [
                        *agent_exec,
                        python,
                        "/pilot/worker.py",
                        arm,
                        cwd,
                        str(info["timeout"]),
                        str(context_window()),
                    ]
                    stamp = dict(
                        task=task,
                        arm=arm,
                        id=uuid.uuid4().hex,
                        model="Qwen3.5-9B",
                        thinking=True,
                        context=context_window(),
                        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                        seed=20261122,
                        base=base,
                        image=image,
                        prompt_sha256=digest(statement),
                        command=command,
                        start=time.time(),
                    )
                    from pilot import manifest

                    stamp["source_manifest"] = manifest()
                    write(destination / "attempt.json", stamp)
                    done = threading.Event()
                    watcher = None
                    if task == SK:
                        watcher = threading.Thread(
                            target=observe_milestones,
                            args=(destination, sandbox, image, base, done),
                            daemon=True,
                        )
                        watcher.start()
                    with (destination / "agent.log").open("w") as f:
                        try:
                            result = subprocess.run(
                                command,
                                timeout=info["timeout"],
                                stdout=f,
                                stderr=subprocess.STDOUT,
                            )
                            stamp["exit_code"] = result.returncode
                        except subprocess.TimeoutExpired:
                            stamp["timeout"] = True
                            # Stop all commands before capturing the final tree.
                            cmd(["docker", "stop", "-t", "1", sandbox])
                            cmd(["docker", "start", sandbox])
                        finally:
                            done.set()
                            if watcher:
                                watcher.join()
                    stamp["seconds"] = time.time() - stamp["start"]
                    if log.exists():
                        with log.open("rb") as f:
                            f.seek(offset)
                            (destination / "inference.jsonl").write_bytes(f.read())
                    usage_file = destination / "inference.jsonl"
                    no_calls = not usage_file.exists() or usage_file.stat().st_size == 0
                    if (
                        stamp.get("exit_code", 0) != 0
                        or no_calls
                        or process.poll() is not None
                    ):
                        stamp["infrastructure_issue"] = (
                            "worker/provider exit or no completed model response; inspect logs before using this trial"
                        )
                    diff = ["git", "diff", "--binary", base, "HEAD"]
                    if task not in DEEP:
                        cmd(["docker", "exec", "-w", cwd, sandbox, "git", "add", "-A"])
                        diff = ["git", "diff", "--cached", "--binary", base]
                    with (destination / "model.patch").open("wb") as f:
                        subprocess.run(
                            ["docker", "exec", "-w", cwd, sandbox, *diff],
                            stdout=f,
                            check=True,
                        )
                    stamp["patch_sha256"] = digest(destination / "model.patch")
                    write(destination / "attempt.json", stamp)
            finally:
                subprocess.run(["docker", "rm", "-f", sandbox], capture_output=True)
