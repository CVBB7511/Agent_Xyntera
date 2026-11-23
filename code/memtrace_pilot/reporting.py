import csv
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing

from common import DEEP, OUT, SK, read, result_directory


def test_command(command):
    # Look at shell statements, excluding heredoc bodies containing Python source.
    # Ambiguous inline programs are left for human review.
    lines = command.splitlines()
    delimiter = None
    for line in lines:
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        marker = re.search(r"""<<-?\s*['"]?(\w+)""", line)
        if marker:
            delimiter = marker[1]
        executable = r"(?:python(?:3(?:\.\d+)?)?\s+(?:-m\s+(?:pytest|unittest)\b|(?:[^\s;&|]+/)?test_[^\s;&|]*\.py\b)|pytest\b|tox\b|ctest\b)"
        if re.search(r"(?:^|&&|\|\||;)\s*" + executable, line):
            return True
    return False


def read_lines(path):
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8")
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]


def matrix():
    yield from (("DeepSWE", t) for t in DEEP)
    selection = OUT / "inputs/evo-selection.json"
    evo = (
        read(selection)
        if selection.exists()
        else [x["instance_id"] for x in read(OUT / "inputs/evo.json")["candidates"]]
    )
    yield from (("SWE-EVO", t) for t in evo)
    yield "SWE-Milestone", SK


def table(name, rows):
    results = result_directory(OUT)
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=results, delete=False
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f.name, results / name)


def report():
    measures = {
        "DeepSWE": ["pass@1"],
        "SWE-EVO": ["Resolved", "Fix", "Apply"],
        "SWE-Milestone": ["Score", "Prec", "Recall", "Resolved"],
    }
    reference = {
        "DeepSWE": [48.7, 66.4],
        "SWE-EVO": [[52.2, 54.1, 95.7], [63, 68.7, 93.5]],
        "SWE-Milestone": [[34.5, 34.7, 45.6, 11.2], [48.3, 46.9, 71.2, 19.4]],
    }
    scores = []
    opportunities = []
    primary_file = OUT / "inputs/evo.json"
    primary = (
        {x["instance_id"] for x in read(primary_file)["candidates"]}
        if primary_file.exists()
        else set()
    )
    for bench, task in matrix():
        pair = {}
        paired_nodes = {}
        for arm in ("native", "memtrace"):
            folder = result_directory(OUT) / "runs" / task / arm
            receipt = (
                read(folder / "grade.json") if (folder / "grade.json").exists() else {}
            )
            attempt = (
                read(folder / "attempt.json")
                if (folder / "attempt.json").exists()
                else {}
            )
            control = OUT / "controls" / task / "status.json"
            status = read(control) if control.exists() else {}
            gpu_file = folder / "gpu.json"
            gpu = read(gpu_file) if gpu_file.exists() else {}
            state = (
                "scored"
                if receipt
                else "attempted_ungraded"
                if attempt
                else "ready"
                if status.get("ready")
                else "infrastructure_pending"
            )
            if attempt.get("infrastructure_issue") or gpu.get("resource_stop"):
                state = "attempt_infrastructure"
            issue = (
                gpu.get("resource_stop")
                or attempt.get("infrastructure_issue")
                or status.get("error", "")
                if state != "scored"
                else ""
            )
            for name in ("launch-blocked.json", "run-error.json", "grade-error.json"):
                pending = folder / name
                if state != "scored" and not issue and pending.exists():
                    issue = read(pending).get("error", "")
            row = dict(
                benchmark=bench,
                task=task,
                arm=arm,
                state=state,
                seconds=attempt.get("seconds"),
                peak_gpu_memory_mib=gpu.get("peak_memory_mib"),
                issue=issue,
                **receipt.get("metrics", {}),
            )
            if bench == "DeepSWE" and receipt:
                reward = receipt.get("reward", {})
                row.update(
                    {
                        key: reward[key]
                        for key in (
                            "f2p_passed",
                            "f2p_total",
                            "p2p_passed",
                            "p2p_total",
                        )
                        if key in reward
                    }
                )
            row["cohort"] = (
                "supplemental"
                if bench == "SWE-EVO" and primary and task not in primary
                else "primary"
            )
            coverage_file = folder / "node-coverage.json"
            if coverage_file.exists():
                coverage = read(coverage_file)
                paired_nodes[arm] = set(coverage["nodes"])
                row.update(
                    scored_nodes=coverage["scored_nodes"],
                    planned_nodes=coverage["planned_nodes"],
                )
                for mid, m in coverage["nodes"].items():
                    scores.append(
                        dict(
                            benchmark=bench,
                            task=task + "/" + mid,
                            arm=arm,
                            state="node_scored",
                            **m,
                        )
                    )
                for mid in coverage["unsubmitted"]:
                    scores.append(
                        dict(
                            benchmark=bench,
                            task=task + "/" + mid,
                            arm=arm,
                            state="node_unsubmitted",
                        )
                    )
                for mid, error in coverage.get("grading_errors", {}).items():
                    scores.append(
                        dict(
                            benchmark=bench,
                            task=task + "/" + mid,
                            arm=arm,
                            state="node_infrastructure",
                            issue=error,
                        )
                    )
            scores.append(row)
            pair[arm] = row
            calls = read_lines(folder / "inference.jsonl")
            actions = read_lines(folder / "trace/actions.jsonl")
            tests = [x for x in actions if test_command(x["command"])]
            audit = {
                "task": task,
                "arm": arm,
                "state": state,
                "cohort": row["cohort"],
                "timed_model_calls": len(calls) if attempt else None,
                "generation_seconds": sum(call["seconds"] for call in calls)
                if calls
                else None,
                "peak_input_tokens": max(
                    (s["usage"]["input_tokens"] for s in calls), default=None
                ),
                "reasoning_tokens": sum(
                    s["usage"]["output_tokens_details"]["reasoning_tokens"]
                    for s in calls
                )
                if calls
                else None,
                "candidate_test_commands": len(tests) if actions else None,
                "candidate_test_seconds": sum(x["seconds"] for x in tests)
                if actions
                else None,
                "review": "pending trajectory review"
                if actions
                else "no observed actions",
            }
            if arm == "native" and calls and attempt.get("seconds") is not None:
                audit["context_refreshes"] = (
                    0  # Stateless Responses provider and unmodified mini 2.4.6 history.
                )
            if arm == "memtrace":
                for db in (folder / "trace").rglob("*"):
                    if db.suffix not in (".db", ".sqlite", ".sqlite3"):
                        continue
                    try:
                        with closing(
                            sqlite3.connect(
                                db.resolve().as_uri() + "?mode=ro", uri=True
                            )
                        ) as connection:
                            count = connection.execute(
                                "select count(*) from context_epochs "
                                "where ordinal > 0 and activated_at is not null"
                            ).fetchone()[0]
                            audit["context_refreshes"] = count
                            break
                    except sqlite3.Error:
                        continue
            review = folder / "audit.json"
            if review.exists():
                audit.update(read(review))
            opportunities.append(audit)
        if all(p["state"] == "scored" for p in pair.values()) and (
            bench != "SWE-Milestone"
            or paired_nodes.get("native") == paired_nodes.get("memtrace")
        ):
            scores.append(
                dict(
                    benchmark=bench,
                    task=task,
                    arm="paired_difference",
                    state="scored",
                    cohort=pair["native"]["cohort"],
                    **{
                        k: pair["memtrace"][k] - pair["native"][k]
                        for k in measures[bench]
                    },
                )
            )
    table("performance.csv", scores)
    table("evidence.csv", opportunities)
    summary = []
    for bench, metrics in measures.items():
        for index, arm in enumerate(("native", "memtrace")):
            for cohort in (
                ("primary", "supplemental") if bench == "SWE-EVO" else ("primary",)
            ):
                rows = [
                    r
                    for r in scores
                    if r["benchmark"] == bench
                    and r["arm"] == arm
                    and r.get("cohort") == cohort
                ]
                good = [r for r in rows if r["state"] == "scored"]
                timed = [r["seconds"] for r in rows if r["seconds"] is not None]
                for i, k in enumerate(metrics):
                    total = sum(r[k] for r in good)
                    ref = reference[bench][index]
                    summary.append(
                        dict(
                            benchmark=bench,
                            arm=arm,
                            cohort=cohort,
                            metric=k,
                            planned_runs=len(rows),
                            attempted=sum(
                                r["state"]
                                in (
                                    "scored",
                                    "attempted_ungraded",
                                    "attempt_infrastructure",
                                )
                                for r in rows
                            ),
                            scored_runs=len(good),
                            sum_of_run_rates=total if good else None,
                            subset_pct=100 * total / len(good) if good else None,
                            mean_seconds=sum(timed) / len(timed) if timed else None,
                            paper_full_DeepSeek_200K_pct=ref
                            if isinstance(ref, float)
                            else ref[i],
                        )
                    )
    table("summary.csv", summary)
