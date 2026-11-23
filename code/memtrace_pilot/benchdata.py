"""Freeze upstream inputs separately from model-visible workspaces."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
from urllib.request import urlopen

from common import DATA, DEEP, EVO, HF, MILE, OUT, SK, cmd, digest, write
from deepswe import describe


def download(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with urlopen(url, timeout=90) as src, temp.open("wb") as dst:
        import shutil

        shutil.copyfileobj(src, dst)
    temp.replace(path)


def checkout(name, source, revision, paths):
    dest = DATA / name
    if not (dest / ".git").exists():
        cmd(
            [
                "git",
                "clone",
                "--no-checkout",
                "--depth=1",
                "--filter=blob:none",
                source,
                dest,
            ]
        )
    if paths:
        cmd(["git", "-C", dest, "sparse-checkout", "set", *paths])
    if subprocess.run(
        ["git", "-C", dest, "cat-file", "-e", revision], capture_output=True
    ).returncode:
        cmd(["git", "-C", dest, "fetch", "--depth=1", "origin", revision])
    cmd(["git", "-C", dest, "checkout", "--detach", revision])
    return dest


def prefix(nodes, edges, ungraded):
    parents = {key: set() for key in nodes}
    for u, v in edges:
        if u in parents and v in parents:
            parents[v].add(u)
    result = []
    while sum(n not in ungraded for n in result) < 3:
        ready = sorted(
            key
            for key, deps in parents.items()
            if key not in result and deps <= set(result)
        )
        if not ready:
            raise ValueError("no three gradeable milestones in DAG")
        result.append(ready[0])
    return result


def prepare():
    import pyarrow.ipc as ipc

    DATA.mkdir(parents=True, exist_ok=True)
    for task in DEEP:
        write(OUT / "inputs" / (task + ".json"), describe(task))
    arrow = DATA / "evo.arrow"
    download(
        f"https://raw.githubusercontent.com/SWE-EVO/SWE-EVO/{EVO}/hf_out/hf_dataset/test/data-00000-of-00001.arrow",
        arrow,
    )
    rows = sorted(
        ipc.open_stream(arrow).read_all().to_pylist(),
        key=lambda row: row["instance_id"],
    )
    if len(rows) != 48:
        raise ValueError("SWE-EVO snapshot must contain 48 instances")
    selected = []
    for row in rows:
        if row["repo"] in [item["repo"] for item in selected]:
            continue
        folder = DATA / "evo" / row["instance_id"]
        write(folder / "row.json", row)
        selected.append(
            {k: row[k] for k in ("instance_id", "repo", "image", "base_commit")}
        )
        if len(selected) == 2:
            break
    write(
        OUT / "inputs/evo.json",
        {"source": EVO, "arrow_sha256": digest(arrow), "candidates": selected},
    )
    checkout(
        "SWE-Milestone", "https://github.com/DeepCommit-ai/SWE-Milestone.git", MILE, []
    )
    # Pin each file to the same official revision; git partial-clone transport
    # on BYL sometimes disconnects when the HF server materializes many blobs.
    ms = DATA / "milestone-files"
    endpoint = "https://huggingface.co/datasets/DeepCommit-ai/SWE-Milestone-data"
    listing = json.load(
        urlopen(
            f"https://huggingface.co/api/datasets/DeepCommit-ai/SWE-Milestone-data/revision/{HF}",
            timeout=30,
        )
    )
    if listing["sha"] != HF:
        raise ValueError("unexpected HF revision")
    files = [
        s["rfilename"]
        for s in listing["siblings"]
        if s["rfilename"].startswith((SK + "/", "config/" + SK))
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda name: download(f"{endpoint}/resolve/{HF}/{name}", ms / name),
                files,
            )
        )
    base = ms / SK

    def table(filename):
        with (base / filename).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def lines(filename):
        p = base / filename
        return (
            {s.strip() for s in p.read_text().splitlines() if s.strip()}
            if p.exists()
            else set()
        )

    records = table("milestones.csv")
    nodes = lines("selected_milestone_ids.txt") or {r["id"] for r in records}
    edges = table("dependencies.csv")
    if (base / "additional_dependencies.csv").is_file():
        edges.extend(table("additional_dependencies.csv"))
    nongraded = lines("non-graded_milestone_ids.txt")
    order = prefix(nodes, [(r["source_id"], r["target_id"]) for r in edges], nongraded)
    refs = dict(
        line.split()
        for line in (DATA / "SWE-Milestone/manifests/digests-v1.0.2.tsv")
        .read_text()
        .splitlines()
        if line.startswith("swe-milestone/")
    )
    images = {
        k: v
        for k, v in refs.items()
        if SK in k
        and k.split("__")[-1].split(":")[0]
        in {"base", "base-offline", *(s.lower() for s in order)}
    }
    write(
        OUT / "inputs/milestone.json",
        {
            "evaluator": MILE,
            "dataset": HF,
            "repository": SK,
            "order": order,
            "graded": [s for s in order if s not in nongraded],
            "images": images,
        },
    )
    checkout("SWE-EVO", "https://github.com/SWE-EVO/SWE-EVO.git", EVO, ["SWE-bench"])


if __name__ == "__main__":
    prepare()
