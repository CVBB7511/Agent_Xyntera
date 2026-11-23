"""Trusted task manifests and isolated copies for test executions."""

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


CONCLUSIVE = {"passed", "assertion_failed"}


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside snapshot: {relative}")
    return candidate


@dataclass(frozen=True)
class Observation:
    epoch: str
    candidate: str
    test: str
    fingerprint: str | None
    status: str
    detail: str
    wall_seconds: float
    cpu_seconds: float | None
    returncode: int | None

    def result(self) -> dict:
        return {key: getattr(self, key) for key in ("epoch", "candidate", "test", "status", "detail")}

    def record(self) -> dict:
        return asdict(self)


class Task:
    def __init__(self, manifest: Path, python: str):
        self.root = manifest.resolve().parent
        self.config = json.loads(manifest.read_text(encoding="utf-8"))
        self.python = shutil.which(python) or str(Path(python).resolve())
        if not Path(self.python).is_file():
            raise ValueError(f"Verifier interpreter not found: {python}")
        self.candidates = self.config["candidates"]
        self.preference = self.config.get("candidate_preference", list(self.candidates))
        if len(self.preference) != len(self.candidates) or set(self.preference) != set(self.candidates):
            raise ValueError("Candidate preference must list all candidates exactly once")
        self.tests = {}
        for spec in self.config["tests"] + self.config["grading_tests"]:
            name = spec["id"]
            if name in self.tests or not spec["argv"]:
                raise ValueError(f"Duplicate ID or empty command: {name}")
            seconds = float(spec.get("estimated_seconds", 1))
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid estimated duration: {name}")
            self.tests[name] = spec
        self.public = [spec["id"] for spec in self.config["tests"]]
        self.grading = [spec["id"] for spec in self.config["grading_tests"]]
        self.initial = [tuple(pair) for pair in self.config["initial_checks"]]
        if not self.initial or not self.grading or len(self.candidates) < 2:
            raise ValueError("Need initial observations, grading tests and two candidates")
        if len(set(self.initial)) != len(self.initial):
            raise ValueError("Duplicate initial observations")
        for candidate, test in self.initial:
            if candidate not in self.candidates or test not in self.public:
                raise ValueError(f"Invalid initial observation: {candidate}/{test}")
        self.readable = set(self.config["readable_sources"])
        self.prefix = self.config["prefix_sources"]
        if not self.prefix or not set(self.prefix) <= self.readable:
            raise ValueError("Prefix sources must be on the read allowlist")

    def snapshot(self, epoch: str) -> Path:
        return inside(self.root, self.config["snapshots"][epoch])

    def overlay(self, candidate: str) -> Path:
        return inside(self.root, self.candidates[candidate]["overlay"])

    def visible_file(self, epoch: str, candidate: str, relative: str) -> Path:
        overlay = inside(self.overlay(candidate), relative)
        return overlay if overlay.is_file() else inside(self.snapshot(epoch), relative)

    def fingerprint(self, epoch: str, candidate: str, test: str) -> str | None:
        spec = self.tests[test]
        deps = spec.get("candidate_dependencies", {}).get(candidate, spec.get("dependencies"))
        if not deps:
            return None
        env_id = self.config.get("epoch_environment_ids", {}).get(epoch, self.config["environment_id"])
        inputs = {
            "id": [candidate, test], "argv": spec["argv"],
            "environment": env_id, "python": self.python,
            "files": [(name, digest(self.visible_file(epoch, candidate, name))) for name in sorted(set(deps))],
        }
        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    def materialize(self, epoch: str, candidate: str, destination: Path) -> None:
        remove_caches = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc")
        shutil.copytree(self.snapshot(epoch), destination, ignore=remove_caches)
        shutil.copytree(self.overlay(candidate), destination, dirs_exist_ok=True, ignore=remove_caches)

    def run(self, epoch: str, candidate: str, test: str, timeout: float) -> Observation:
        fingerprint = self.fingerprint(epoch, candidate, test)
        with tempfile.TemporaryDirectory(prefix="agent-v00-") as tmp:
            root = Path(tmp) / "workspace"
            self.materialize(epoch, candidate, root)
            argv = [self.python if value == "{python}" else value for value in self.tests[test]["argv"]]
            env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "LANG") if key in os.environ}
            env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                        "HOME": tmp, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp})
            before = child_cpu()
            start = time.monotonic()
            with tempfile.TemporaryFile(mode="w+b") as output:
                try:
                    with subprocess.Popen(
                        argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
                    ) as process:
                        try:
                            returncode = process.wait(timeout=timeout)
                            content = output_tail(output)
                            status, detail = parse_result(returncode, content)
                        except subprocess.TimeoutExpired:
                            try:
                                if os.name == "posix":
                                    os.killpg(process.pid, signal.SIGKILL)
                                else:
                                    process.kill()
                            except ProcessLookupError:
                                pass
                            returncode = process.wait()
                            status, detail = "timeout", output_tail(output)
                except OSError as exc:
                    returncode = None
                    status, detail = "execution_error", str(exc)
            elapsed = time.monotonic() - start
            after = child_cpu()
        return Observation(epoch, candidate, test, fingerprint, status, detail[:2000], elapsed,
                           None if before is None else max(0.0, after - before), returncode)


def output_tail(output) -> str:
    size = output.tell()
    output.seek(max(0, size - 4096))
    return output.read().decode("utf-8", errors="replace").strip()


def parse_result(returncode: int, content: str) -> tuple[str, str]:
    if returncode != 0:
        return "execution_error", content
    try:
        result = json.loads(content.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return "invalid_result", content
    if (not isinstance(result, dict) or not isinstance(result.get("status"), str)
            or result["status"] not in CONCLUSIVE | {"not_collected"}):
        return "invalid_result", content
    return result["status"], str(result.get("detail", ""))


def child_cpu() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime
