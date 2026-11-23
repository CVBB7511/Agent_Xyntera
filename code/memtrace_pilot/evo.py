import json
import os
import re
import subprocess
import uuid
import fcntl

from common import (
    DATA,
    OUT,
    PREFIX,
    SRC,
    cmd,
    digest,
    image_manifest,
    private_pull,
    read,
    release_cached_ref,
    write,
)

DASK = "dask__dask_2022.9.2_2022.10.0"
DASK_WHEEL = "pandas-1.5.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
DASK_WHEEL_SHA256 = "7a0a56cef15fd1586726dace5616db75ebcfec9179a3a55e78f72c5639fa2a23"


def grade(row, patch, destination, info):
    """Keep the upstream scoring program intact; supply its SWE-agent predictions format."""
    image = private_pull(info)
    # The official harness expects this source tag to resolve in its private daemon.
    cmd(["docker", "tag", image, row["image"]])
    evaluator = DATA / "SWE-EVO/SWE-bench"
    run_id = "evo-" + uuid.uuid4().hex
    path = destination / run_id
    write(path / "output_final" / (row["instance_id"] + ".json"), row)
    write(
        path / run_id / "preds.json",
        {row["instance_id"]: {"model_patch": patch.read_text()}},
    )
    env = os.environ | {
        "PYTHONPATH": str(evaluator),
        "PYTHONNOUSERSITE": "1",
        "PILOT_EVO_IMAGE": image,
        "PILOT_EVO_INSTANCE": json.dumps(
            {
                key: row.get(key)
                for key in (
                    "repo",
                    "base_commit",
                    "environment_setup_commit",
                    "instance_id",
                    "image",
                )
            }
        ),
    }
    pinned_pandas = (
        row["instance_id"] == DASK and (DATA / "dask-wheels" / DASK_WHEEL).exists()
    )
    if pinned_pandas:
        wheel = DATA / "dask-wheels" / DASK_WHEEL
        if digest(wheel) != DASK_WHEEL_SHA256:
            raise ValueError("Dask pinned pandas wheel hash mismatch")
        env["PILOT_EVO_WHEELS"] = str(wheel.parent)
    command = [
        str(PREFIX / "bin/python"),
        str(SRC / "evo_evaluator.py"),
        str(evaluator / "evaluate_instance.py"),
        "--instance",
        row["instance_id"],
        "--trajectories_path",
        str(path / run_id),
        "--scaffold",
        "SWE-agent",
        "--max_workers",
        "1",
    ]
    with (path / "grader.log").open("w") as f:
        subprocess.run(
            command,
            cwd=path,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=2700,
        )
    stdout = (path / "grader.log").read_text()
    metrics = {}
    for tag, key in [
        ("Resolved rate", "Resolved"),
        ("Fix rate", "Fix"),
        ("Applied rate", "Apply"),
    ]:
        match = re.search(r"\[" + tag + r"\]\s*=\s*([\d.]+)%", stdout)
        if not match:
            raise ValueError(f"official evaluator missing {tag}")
        metrics[key] = float(match[1]) / 100
    logs = list(path.rglob("test_output.txt"))
    if len(logs) != 1 or logs[0].stat().st_size == 0:
        raise ValueError("official per-instance test output missing")
    if pinned_pandas and "Successfully installed pandas-1.5.3" not in logs[0].read_text(
        errors="replace"
    ):
        raise ValueError("Dask evaluator did not install the pinned pandas 1.5.3")
    receipt = {
        "metrics": metrics,
        "image": image,
        "patch_sha256": digest(patch),
        "official_output": str(path.relative_to(OUT)),
        "command": command,
    }
    if pinned_pandas:
        receipt["pandas_wheel_sha256"] = DASK_WHEEL_SHA256
    write(destination / "grade.json", receipt)
    return receipt


def regrade_dask():
    row = read(DATA / "evo" / DASK / "row.json")
    info = read(OUT / "inputs" / (DASK + ".json"))["image"]
    controls_root = OUT / "controls" / DASK
    for arm in ("native", "memtrace"):
        folder = OUT / "runs" / DASK / arm
        receipt = folder / "grade.json"
        if (
            receipt.exists()
            and read(receipt).get("pandas_wheel_sha256") != DASK_WHEEL_SHA256
        ):
            receipt.rename(folder / ("grade-invalid-" + uuid.uuid4().hex + ".json"))
            write(
                folder / "grade-error.json",
                {"error": "pandas installation timed out in the original evaluator"},
            )
    try:
        for label, resolved in (("empty", 0), ("gold", 1)):
            receipt = grade(
                row,
                controls_root / (label + ".patch"),
                controls_root / ("pinned-" + label),
                info,
            )
            reports = list((OUT / receipt["official_output"]).rglob("report.json"))
            if len(reports) != 1 or receipt["metrics"]["Resolved"] != resolved:
                raise ValueError(
                    f"Dask {label} pinned dependency control had an unexpected verdict"
                )
            status = read(reports[0])[DASK]["tests_status"]
            if status["PASS_TO_PASS"]["failure"]:
                raise ValueError(
                    f"Dask {label} pinned control regressed baseline tests"
                )
        for arm in ("native", "memtrace"):
            folder = OUT / "runs" / DASK / arm
            receipt = folder / "grade.json"
            if not receipt.exists():
                grade(row, folder / "model.patch", folder, info)
    finally:
        release_image(DASK)


def release_image(task):
    info_file = OUT / "inputs" / (task + ".json")
    if not info_file.exists():
        return
    ref = read(info_file)["image"]["ref"]
    result = subprocess.run(
        ["docker", "image", "inspect", ref], capture_output=True, text=True
    )
    if result.returncode:
        return
    tags = json.loads(result.stdout)[0]["RepoTags"] or []
    for tag in tags:
        if tag.startswith(("sweb.eval.", "sweb.env.", ref.split("@")[0] + ":")):
            release_cached_ref(tag)
    release_cached_ref(ref)


def controls():
    with (OUT / "evo-controls.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _select()


def _select():
    import pyarrow.ipc as ipc

    rows = sorted(
        ipc.open_stream(DATA / "evo.arrow").read_all().to_pylist(),
        key=lambda r: r["instance_id"],
    )
    seen = set()
    primary = [r["instance_id"] for r in read(OUT / "inputs/evo.json")["candidates"]]
    qualified = []
    reviewed = []
    for row in rows:
        if row["repo"] in seen:
            continue
        task = row["instance_id"]
        path = OUT / "controls" / task
        seen.add(row["repo"])
        reviewed.append(task)
        loaded_image = False
        try:
            info_file = OUT / "inputs" / (task + ".json")
            image = (
                read(info_file)["image"]
                if info_file.exists()
                else image_manifest(row["image"] + ":latest")
            )
            write(
                OUT / "inputs" / (task + ".json"),
                dict(id=task, base=row["base_commit"], image=image, timeout=14400),
            )
            write(DATA / "evo" / task / "row.json", row)
            path.mkdir(parents=True, exist_ok=True)
            for label, content in [("empty", ""), ("gold", row["patch"])]:
                patch = path / (label + ".patch")
                patch.write_text(content)
                receipt_path = path / label / "grade.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                else:
                    loaded_image = True
                    receipt = grade(row, patch, path / label, image)
                if (
                    receipt["patch_sha256"] != digest(patch)
                    or receipt["image"] != image["ref"]
                ):
                    raise ValueError("control provenance mismatch")
                if label == "empty":
                    reports = list(
                        (OUT / receipt["official_output"]).rglob("report.json")
                    )
                    if len(reports) != 1:
                        raise ValueError(
                            "missing per-instance report to check baseline eligibility"
                        )
                    f2p = read(reports[0])[task]["tests_status"]["FAIL_TO_PASS"]
                    if not f2p["failure"]:
                        if not f2p["success"]:
                            raise ValueError("no F2P outcomes in control")
                        break
                if label == "gold" and not receipt["metrics"]["Resolved"]:
                    raise ValueError("reference submission did not resolve task")
            else:
                write(path / "status.json", {"ready": True})
                qualified.append(task)
            if task not in qualified:
                write(
                    path / "status.json",
                    {"ready": False, "ineligible": "all F2P pass before changes"},
                )
        except Exception as exc:
            write(path / "status.json", {"ready": False, "error": str(exc)})
        # Screening can span several large images. The frozen ref and receipts
        # let each task reload its image when the pair starts.
        if loaded_image:
            release_image(task)
        supplements = [t for t in qualified if t not in primary]
        write(OUT / "inputs/evo-selection.json", primary + supplements)
        write(
            OUT / "inputs/evo-screening.json",
            {
                "primary": primary,
                "reviewed": reviewed,
                "qualified": qualified,
                "supplemental": supplements,
                "rule": "first distinct repository by instance ID; two qualified pairs",
            },
        )
        if len(qualified) == 2:
            break
