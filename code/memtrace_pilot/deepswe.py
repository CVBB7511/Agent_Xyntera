import json
import shutil
import subprocess
import tomllib
import uuid
from common import (
    DEEP,
    ROOT,
    OUT,
    cmd,
    digest,
    image_manifest,
    private_pull,
    read,
    write,
)


def describe(task):
    repo = ROOT / "deep-swe"
    sha = cmd(["git", "-C", repo, "rev-parse", "HEAD"])
    if sha != "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea":
        raise ValueError("DeepSWE snapshot changed")
    base = repo / "tasks" / task
    spec = tomllib.loads((base / "task.toml").read_text())
    return dict(
        id=task,
        source=sha,
        base=spec["metadata"]["base_commit_hash"],
        image=image_manifest(spec["environment"]["docker_image"]),
        task_hashes={
            str(p.relative_to(base)): digest(p)
            for p in sorted(base.rglob("*"))
            if p.is_file()
        },
        timeout=min(14400, int(spec["agent"]["timeout_sec"])),
    )


def grade(task, patch, directory, info):
    ref = private_pull(info["image"])
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / ("official-" + uuid.uuid4().hex)
    (logs / "artifacts").mkdir(parents=True)
    shutil.copyfile(patch, logs / "artifacts/model.patch")
    tests = ROOT / "deep-swe/tasks" / task / "tests"
    name = "pilot-judge-" + uuid.uuid4().hex
    try:
        args = [
            "docker",
            "run",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{tests}:/tests:ro",
            "-v",
            f"{logs}:/logs",
            ref,
            "bash",
            "/tests/test.sh",
        ]
        with (logs / "stdout.log").open("w") as f:
            subprocess.run(
                args, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1800
            )
        reward = read(logs / "verifier/reward.json")
        score = reward.get("reward")
        if score not in (0, 1):
            raise ValueError(f"unexpected official reward keys {sorted(reward)}")
        receipt = {
            "metrics": {"pass@1": score},
            "patch_sha256": digest(patch),
            "reward": reward,
            "official_output": str(logs.relative_to(OUT)),
            "image": ref,
        }
        write(directory / "grade.json", receipt)
        return receipt
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def controls(task, info):
    folder = OUT / "controls" / task
    folder.mkdir(parents=True, exist_ok=True)
    empty = folder / "empty.patch"
    empty.write_bytes(b"")
    for label, patch, want in [
        ("empty", empty, 0),
        ("gold", ROOT / "deep-swe/tasks" / task / "solution/solution.patch", 1),
    ]:
        path = folder / label
        old = path / "grade.json"
        receipt = read(old) if old.exists() else {}
        if receipt.get("image") != info["image"]["ref"] or receipt.get(
            "patch_sha256"
        ) != digest(patch):
            receipt = grade(task, patch, path, info)
        if receipt["metrics"]["pass@1"] != want:
            raise ValueError(f"{label} control gave unexpected reward")


if __name__ == "__main__":
    for task in DEEP:
        try:
            manifest = OUT / "inputs" / (task + ".json")
            if manifest.exists():
                info = read(manifest)
                repo = ROOT / "deep-swe"
                task_dir = repo / "tasks" / task
                if cmd(["git", "-C", repo, "rev-parse", "HEAD"]) != info["source"]:
                    raise ValueError("DeepSWE snapshot changed")
                if any(
                    digest(task_dir / p) != h for p, h in info["task_hashes"].items()
                ):
                    raise ValueError("frozen DeepSWE task file changed")
            else:
                info = describe(task)
                write(manifest, info)
            controls(task, info)
            status = {"ready": True}
        except Exception as exc:
            status = {"ready": False, "error": str(exc)}
        write(OUT / "controls" / task / "status.json", status)
        print(task, json.dumps(status), flush=True)
