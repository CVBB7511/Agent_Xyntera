"""Use the official DAG manager, immutable tag snapshots, evaluator and scoring functions."""

import subprocess
import sys
import uuid
from common import (
    DATA,
    OUT,
    SK,
    cmd,
    image_manifest,
    private_pull,
    read,
    release_cached_ref,
    result_directory,
    write,
)

WORK = DATA / "milestone-files" / SK
UPSTREAM = DATA / "SWE-Milestone"
sys.path.insert(0, str(UPSTREAM))


def setup_runner(mid, folder, container, image):
    from harness.e2e.run_milestone import MilestoneRunner

    target = MilestoneRunner(
        workspace_root=WORK,
        milestone_id=mid,
        srs_path=WORK / "srs" / mid / "SRS.md",
        output_dir=folder,
        image_name=image,
        trial_name="pilot",
    )
    target.container_name = container
    target.container_setup.container_name = container
    return target


def images():
    info = read(OUT / "inputs/milestone.json")
    manifest = OUT / "inputs/milestone-images.json"
    previous = read(manifest) if manifest.exists() else {}
    frozen = {}
    for tag, ref in info["images"].items():
        key = tag.split("__")[-1].split(":")[0]
        if key == "base":
            continue
        saved = previous.get(key, {})
        item = (
            saved
            if saved.get("ref") == ref and saved.get("tag") == tag
            else image_manifest(ref)
        )
        if item["ref"] != ref:
            raise ValueError("milestone digest mismatch")
        frozen[key] = item | {"tag": tag}
    write(manifest, frozen)
    return frozen


def capture(mid, folder, container, image, base):
    runner = setup_runner(mid, folder, container, image)
    runner._snapshot_baseline_commit = base
    runner.container_setup.workdir = "/testbed"
    folder.mkdir(parents=True, exist_ok=True)
    snap = runner._extract_snapshot()
    write(folder / "snapshot.json", {"path": str(snap), "base": base})
    return snap


def snapshot_user(container):
    # The official capture helper reads /testbed as the image-local fakeroot account.
    cmd(
        [
            "docker",
            "exec",
            "--user",
            "root",
            container,
            "sh",
            "-ec",
            "id fakeroot >/dev/null 2>&1 || useradd -m fakeroot; git config --system --add safe.directory /testbed",
        ]
    )


def score(mid, folder):
    from pathlib import Path

    runner = setup_runner(mid, folder, "no-live-agent", "unused")
    snap = Path(read(folder / "snapshot.json")["path"])
    outcome = runner._run_evaluation(snap)
    raw = outcome.to_dict()
    write(folder / "official.json", raw)
    return milestone_metrics(raw, mid)


def milestone_metrics(raw, mid):
    from harness.e2e.collect_results import (
        calculate_precision_recall,
        calculate_score_reliable,
        is_infra_invalid,
        is_resolved,
    )

    if raw.get("error") or is_infra_invalid(raw):
        raise RuntimeError(f"infrastructure-invalid milestone evaluation {mid}")
    prec, rec = calculate_precision_recall(raw)
    m = {
        "Score": calculate_score_reliable(raw),
        "Prec": prec,
        "Recall": rec,
        "Resolved": int(is_resolved(raw)),
    }
    if any(x is None for x in m.values()):
        raise ValueError("official milestone score unavailable")
    return m


def controls():
    folder = OUT / "controls" / SK
    try:
        refs = images()
        data = read(OUT / "inputs/milestone.json")
        metadata = {m["id"]: m for m in read(WORK / "metadata.json")["milestones"]}
        warnings = []
        for mid in data["graded"]:
            empty_file = folder / mid / "empty" / "grade.json"
            gold_file = folder / mid / "gold" / "grade.json"
            image = refs[mid.lower()]
            for receipt in (empty_file, gold_file):
                cached = receipt.with_name("official.json")
                if not receipt.exists() and cached.exists():
                    try:
                        metrics = milestone_metrics(read(cached), mid)
                    except (ValueError, RuntimeError):
                        continue
                    write(receipt, {"metrics": metrics})
            if not empty_file.exists() or not gold_file.exists():
                ref = private_pull(image)
                cmd(["docker", "tag", ref, image["tag"]])
                name = "pilot-ms-check-" + uuid.uuid4().hex
                try:
                    cmd(
                        [
                            "docker",
                            "run",
                            "-d",
                            "--name",
                            name,
                            "--network",
                            "none",
                            "-w",
                            "/testbed",
                            ref,
                            "sleep",
                            "infinity",
                        ]
                    )
                    snapshot_user(name)
                    rec = metadata[mid]
                    for label, key in [
                        ("empty", "commit_sha_start"),
                        ("gold", "commit_sha_end"),
                    ]:
                        path = folder / mid / label
                        receipt = path / "grade.json"
                        if receipt.exists():
                            continue
                        cmd(
                            [
                                "docker",
                                "exec",
                                name,
                                "git",
                                "-c",
                                "safe.directory=/testbed",
                                "tag",
                                "-f",
                                "agent-impl-" + mid,
                                rec[key],
                            ]
                        )
                        capture(mid, path, name, ref, rec["commit_sha_start"])
                        metrics = score(mid, path)
                        write(receipt, {"metrics": metrics})
                finally:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                    release_cached_ref(image["tag"])
                    release_cached_ref(ref)
            empty, gold = read(empty_file)["metrics"], read(gold_file)["metrics"]
            if empty["Resolved"] or gold["Score"] <= empty["Score"]:
                raise RuntimeError(f"{mid} reference did not improve official Score")
            if not gold["Resolved"]:
                warnings.append({"milestone": mid, "reference": gold})
        if len(warnings) == len(data["graded"]):
            raise RuntimeError("no milestone had a strictly resolved positive control")
        image = refs["base-offline"]
        frozen = OUT / "inputs" / (SK + ".json")
        if frozen.exists() and read(frozen)["image"]["ref"] == image["ref"]:
            write(
                folder / "status.json", {"ready": True, "reference_warnings": warnings}
            )
            return
        ref = private_pull(image)
        cmd(["docker", "tag", ref, image["tag"]])
        base = cmd(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-w",
                "/testbed",
                ref,
                "git",
                "-c",
                "safe.directory=/testbed",
                "rev-parse",
                "HEAD",
            ]
        )
        write(
            frozen,
            {"id": SK, "base": base, "image": image, "timeout": 14400},
        )
        write(folder / "status.json", {"ready": True, "reference_warnings": warnings})
        release_cached_ref(image["tag"])
        release_cached_ref(ref)
    except Exception as exc:
        import traceback

        write(
            folder / "status.json",
            {"ready": False, "error": str(exc), "traceback": traceback.format_exc()},
        )


def run(arm, *, time_limit=None):
    from trial import attempt

    # All three nodes in the frozen scikit-learn prefix are initially open.
    from harness.e2e.dag import DAGManager

    data = read(OUT / "inputs/milestone.json")
    ids = OUT / "inputs/milestone-ids.txt"
    ids.write_text("\n".join(data["order"]) + "\n")
    dag = DAGManager(WORK / "dependencies.csv", selected_ids_file=ids)
    if set(dag.get_next_runnable()) != set(data["order"]):
        raise RuntimeError("subset requires dynamic downstream unlock")
    prompt = "Complete the following available milestones in one workspace. Commit each change and create its indicated submission tag.\n"
    for mid in data["order"]:
        prompt += (
            "\nSubmission tag: agent-impl-"
            + mid
            + "\n"
            + (WORK / "srs" / mid / "SRS.md").read_text()
        )
    info = read(OUT / "inputs" / (SK + ".json"))
    if time_limit is not None:
        info["timeout"] = min(info["timeout"], time_limit)
    attempt(SK, arm, info, prompt, "/testbed")


def grade_saved(arm):
    dest = result_directory(OUT) / "runs" / SK / arm
    attempt = dest / "attempt.json"
    if not attempt.exists() or "seconds" not in read(attempt):
        return
    data = read(OUT / "inputs/milestone.json")
    refs = read(OUT / "inputs/milestone-images.json")
    # scikit-learn's evaluator uses its own official images. The source
    # archive and integrity sidecar are already saved before grading.
    release_cached_ref(refs["base-offline"]["tag"])
    release_cached_ref(refs["base-offline"]["ref"])
    rows = {}
    missing = []
    errors = {}
    for mid in data["graded"]:
        path = dest / "nodes" / mid
        receipt = path / "grade.json"
        if receipt.exists():
            rows[mid] = read(receipt)["metrics"]
        elif (path / "snapshot.json").exists():
            ref = refs[mid.lower()]
            try:
                pinned = private_pull(ref)
                cmd(["docker", "tag", pinned, ref["tag"]])
                rows[mid] = score(mid, path)
                write(receipt, {"metrics": rows[mid], "image": pinned})
            except Exception as exc:
                errors[mid] = str(exc)
            finally:
                release_cached_ref(ref["tag"])
                release_cached_ref(ref["ref"])
        elif (path / "capture-error.json").exists():
            errors[mid] = read(path / "capture-error.json")["error"]
        else:
            missing.append(mid)
    coverage = {
        "nodes": rows,
        "unsubmitted": missing,
        "planned_nodes": len(data["graded"]),
        "scored_nodes": len(rows),
        "grading_errors": errors,
    }
    write(dest / "node-coverage.json", coverage)
    if errors:
        raise RuntimeError(
            f"milestone evaluator or capture failed for {', '.join(errors)}"
        )
    if rows:
        write(
            dest / "grade.json",
            {
                **coverage,
                "metrics": {
                    k: sum(m[k] for m in rows.values()) / len(rows)
                    for k in next(iter(rows.values()))
                },
            },
        )


if __name__ == "__main__":
    mode, mid, folder, container, image, base = sys.argv[1:]
    from pathlib import Path

    if mode == "capture":
        capture(mid, Path(folder), container, image, base)
    elif mode == "truncate":
        runner = setup_runner(mid, Path(folder), container, image)
        snapshot_user(container)
        # The rootless image's source is owned by container root. Set the
        # baseline and trim refs as that owner, without copying the entire tree.
        freeze = """set -eu
git checkout -q -B main "$1"
git remote | while IFS= read -r name; do git remote remove "$name"; done
git for-each-ref --format='%(refname)' |
    awk '$0 != "refs/heads/main" { print "delete " $0 }' | git update-ref --no-deref --stdin
git reflog expire --expire=now --all
git gc --prune=now --quiet
test "$(git rev-parse HEAD)" = "$1"
test "$(git for-each-ref --format='%(refname)')" = refs/heads/main
"""
        cmd(
            [
                "docker",
                "exec",
                "--user",
                "root",
                "-w",
                "/testbed",
                container,
                "bash",
                "-c",
                freeze,
                "pilot-baseline",
                base,
            ],
            timeout=300,
        )
        for milestone in read(OUT / "inputs/milestone.json")["order"]:
            staged = (
                runner
                if milestone == mid
                else setup_runner(milestone, Path(folder), container, image)
            )
            staged._apply_test_masking()
