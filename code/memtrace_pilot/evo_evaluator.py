"""Run the upstream grader with frozen dependency files from its own image."""

import json
import os
from pathlib import PurePosixPath
import runpy
import subprocess
import sys
from urllib.parse import urlsplit
from unittest.mock import patch
import hashlib
from contextlib import ExitStack


def cached_evaluation_image(collection, repository, tag, *, source, pinned):
    """Resolve the grader's source alias to the already staged digest."""
    if tag not in (None, "latest") or repository not in (source, f"{source}:latest"):
        return None
    frozen = collection.get(pinned)
    print(f"[local image] {source} -> {pinned}", flush=True)
    return frozen


def main():
    import requests
    from docker.models.containers import ContainerCollection
    from docker.models.images import ImageCollection

    evaluator, *args = sys.argv[1:]
    row = json.loads(os.environ["PILOT_EVO_INSTANCE"])
    pinned_image = os.environ["PILOT_EVO_IMAGE"]
    wheels = os.environ.get("PILOT_EVO_WHEELS")
    revisions = {row.get("base_commit"), row.get("environment_setup_commit")}
    upstream_get = requests.get
    git = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-w",
        "/testbed",
        "--entrypoint",
        "git",
        pinned_image,
        "-c",
        "safe.directory=/testbed",
    ]

    def get(url, *a, **kw):
        parsed = urlsplit(url)
        source = parsed.path.lstrip("/").split("/", 3)
        if (
            parsed.netloc != "raw.githubusercontent.com"
            or len(source) != 4
            or "/".join(source[:2]) != row["repo"]
            or source[2] not in revisions
        ):
            return upstream_get(url, *a, **kw)
        revision, filename = source[2:]
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe environment file path")
        result = subprocess.run(
            [*git, "show", f"{revision}:{filename}"], capture_output=True, timeout=120
        )
        if result.returncode:
            # Preserve upstream fallback when the benchmark image lacks the commit
            # or the repository uses another supported environment filename.
            kw.setdefault("timeout", 20)
            return upstream_get(url, *a, **kw)
        response = requests.Response()
        response.status_code = 200
        response.encoding = "utf-8"
        response.url = url
        response._content = result.stdout
        print(
            f"Frozen environment input {url} sha256={hashlib.sha256(result.stdout).hexdigest()}",
            flush=True,
        )
        return response

    create = ContainerCollection.create
    pull = ImageCollection.pull

    def pull_evaluator_image(collection, repository, tag=None, **kwargs):
        pinned = cached_evaluation_image(
            collection,
            repository,
            tag,
            source=row["image"],
            pinned=pinned_image,
        )
        if pinned is not None:
            return pinned
        return pull(collection, repository, tag=tag, **kwargs)

    def create_evaluator(collection, image, *a, **kw):
        if (kw.get("name") or "").startswith(f"sweb.eval.{row['instance_id']}."):
            # Pin the evaluator even if upstream skipped retagging an existing alias.
            image = pinned_image
            if wheels:
                kw["volumes"] = (kw.get("volumes") or {}) | {
                    wheels: {"bind": "/pilot-wheels", "mode": "ro"},
                }
                kw["environment"] = (kw.get("environment") or {}) | {
                    "PIP_NO_INDEX": "1",
                    "PIP_FIND_LINKS": "/pilot-wheels",
                }
        return create(collection, image, *a, **kw)

    sys.argv = [evaluator, *args]
    with ExitStack() as patches:
        patches.enter_context(patch.object(requests, "get", get))
        patches.enter_context(
            patch.object(ImageCollection, "pull", pull_evaluator_image)
        )
        patches.enter_context(
            patch.object(ContainerCollection, "create", create_evaluator)
        )
        runpy.run_path(evaluator, run_name="__main__")


if __name__ == "__main__":
    main()
