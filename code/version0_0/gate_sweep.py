"""Compare verification schedules with a fixed submission proposal and test budget."""

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiment import Limits, Run, prepare_reference
from model import ScriptedModel
from run import write_json
from task import Task


SCHEDULES = ("guarded-eager", "guarded")


def measure(case: Path, budgets: list[int], output: Path, python: str) -> list[dict]:
    task = Task(case, python)
    if "candidate_preference" not in task.config:
        raise ValueError(f"No candidate priority declared in {case}")
    limits = Limits(8192, 6144, 1024, 110_000, 0, 90, 20, 8)
    initial, reference = prepare_reference(task, limits)
    observed = {(o.candidate, o.test): o.status for o in initial}
    proposal = next((c for c in task.preference if all(observed.get((c, t)) == "passed" for t in task.public)), None)
    if proposal is None:
        raise ValueError(f"No publicly verified E0 proposal in {case}")
    provenance = task.config["provenance"]
    defect = f"{provenance['project']}/{provenance['bug']}"
    rows = []
    for budget in budgets:
        for schedule in SCHEDULES:
            path = output / f"{budget}-{schedule}"
            run = Run(task, schedule, ScriptedModel(), replace(limits, max_tests=budget), path, initial)
            run.phase, run.epoch = 2, "E1"
            selected = run.check_submission(proposal)
            grading = [] if selected is None else [task.run("E1", selected, t, limits.test_timeout) for t in task.grading]
            passed = bool(grading) and all(o.status == "passed" for o in grading)
            preferred = run.metrics(reference)["preferred_viable"]
            row = {"case": task.config["id"], "defect": defect, "budget": budget, "schedule": schedule,
                   "gate": run.submission_gate, "preferred_viable": preferred,
                   "passed": passed, "selected_preferred": passed and selected == preferred,
                   "execution_tests": [o.record() for o in run.executions], "grading": [o.record() for o in grading]}
            write_json(path / "summary.json", row)
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["budget"], row["schedule"]][row["defect"]].append(row)
    return [{"budget": budget, "schedule": schedule, "defects": len(defects),
             "cases": sum(map(len, defects.values())),
             "mean_tests_per_defect": mean(mean(r["gate"]["physical_tests"] for r in group) for group in defects.values()),
             "mean_acceptance_per_defect": mean(mean(r["passed"] for r in group) for group in defects.values()),
             "mean_preferred_per_defect": mean(mean(r["selected_preferred"] for r in group) for group in defects.values())}
            for (budget, schedule), defects in sorted(groups.items())]


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", type=Path, nargs="+", required=True)
    cli.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 12])
    cli.add_argument("--test-python", default=sys.executable)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if min(args.budgets) < 0 or len(args.budgets) != len(set(args.budgets)):
        cli.error("Use distinct non-negative budgets")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(args.case, start=1):
        folder = args.output / f"case-{index:03d}"
        folder.mkdir()
        rows.extend(measure(case, args.budgets, folder, args.test_python))
        print(f"Completed {case.parent.name}", flush=True)
        write_json(args.output / "results.json", {"cases": rows, "defect_weighted": summarize(rows)})


if __name__ == "__main__":
    main()
