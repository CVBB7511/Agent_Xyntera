from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/mnt/sata1/beihang-cuda_backup")
DATA = ROOT / "vendor" / "pilot-data"
OUT = ROOT / "results" / "memtrace-pilot"
LEGACY_CONTEXT_WINDOW = 131_072
CONTEXT_WINDOWS = (LEGACY_CONTEXT_WINDOW, 262_144)
OUTPUT_RESERVE = 8192
SRC = ROOT / "code" / "memtrace_pilot"
MODEL = ROOT / "models" / "Qwen3.5-9B"
PREFIX = ROOT / ".conda" / "memtrace-pilot"
DEEP = ("adaptix-name-mapping-aliases", "aiomonitor-task-snapshots-diff")
EVO = "9b83d5af943ba7a17567336f5b18239f73960219"
MILE = "17a8f1593e172e26b36cea15e2b30fb9536c93f5"
HF = "3500478e740d42583277c2658aa7729eb36e10f4"
SK = "scikit-learn_scikit-learn_1.5.2_1.6.0"


def context_window():
    limit = int(os.environ.get("PILOT_CONTEXT_WINDOW", CONTEXT_WINDOWS[-1]))
    if limit not in CONTEXT_WINDOWS:
        raise ValueError(f"unsupported pilot context window: {limit}")
    return limit


def result_directory(root):
    limit = context_window()
    return root if limit == LEGACY_CONTEXT_WINDOW else root / f"context-{limit}"


def cmd(args, **kwargs):
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(f.name, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def image_manifest(ref):
    details = json.loads(cmd(["docker", "manifest", "inspect", "-v", ref], timeout=45))
    choices = details if isinstance(details, list) else [details]
    item = next(
        x
        for x in choices
        if x["Descriptor"].get("platform", {}).get("architecture") == "amd64"
    )
    raw = base64.b64decode(item["Raw"])
    sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    if sha != item["Descriptor"]["digest"]:
        raise ValueError("image descriptor digest mismatch")
    name = ref.split("@")[0]
    if ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]
    return {
        "ref": name + "@" + sha,
        "compressed_bytes": sum(x["size"] for x in json.loads(raw)["layers"]),
    }


def private_pull(info):
    store = cmd(["docker", "info", "--format", "{{.DockerRootDir}}"])
    if Path(store) != ROOT / ".pilot-docker/data":
        raise ValueError("Docker writes require the private project daemon")
    ref = info["ref"]
    if subprocess.run(
        ["docker", "image", "inspect", ref], capture_output=True
    ).returncode:
        estimate = info["compressed_bytes"] * 4 + 2 * 2**30
        if shutil.disk_usage(ROOT).free < estimate:
            if ref.startswith(("xingyaoww/", "hyd2apse/")):
                from capacity import installed_layers, layer_sizes

                present = installed_layers()
                sizes = layer_sizes(ref)
                estimate = (
                    sum(v for k, v in sizes.items() if k not in present)
                    + info["compressed_bytes"]
                    + 2**30
                )
            free = shutil.disk_usage(ROOT).free
            if free < estimate:
                raise OSError(
                    f"image staging peak estimated {estimate} bytes; available {free} bytes"
                )
        subprocess.run(["docker", "pull", ref], check=True, timeout=900)
    return ref


def release_cached_ref(ref):
    """Release a specific reproducible ref in our daemon after scoring."""
    if cmd(["docker", "info", "--format", "{{.DockerRootDir}}"]) != str(
        ROOT / ".pilot-docker/data"
    ):
        raise ValueError("refusing to change a shared image store")
    subprocess.run(["docker", "image", "rm", ref], capture_output=True)
