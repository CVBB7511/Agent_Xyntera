"""Aggregate case-level results and emit matched eager/selective contrasts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = (
    "passed", "selected_preferred_viable", "physical_tests", "repeat_executions",
    "reopening_opportunities_e1", "reopening_checks_missed_e1",
    "independent_blocker_opportunities_e1", "independent_blocker_retests_e1",
    "test_cpu_seconds", "generation_seconds", "run_wall_seconds",
    "blocked_verification_requests", "unchanged_e1_retested", "continuation_physical_tests",
)

FOLLOWUP_PAIRS = (("eager", "eager-visible"), ("selective", "guarded"), ("guarded", "guarded-pruned"))


def summarize(inputs: list[Path]) -> dict:
    by_arm = defaultdict(list)
    pairs = []
    seen = set()
    rejected = []
    shared_boundaries = {}
    notice_pairs = []
    scheduler_pairs = []
    for path in inputs:
        for case in json.loads(path.read_text(encoding="utf-8")):
            if "results" not in case:
                rejected.append(case)
                continue
            case_id = case["case"]
            if case_id in seen:
                raise ValueError(f"Repeated case in pilot report: {case_id}")
            seen.add(case_id)
            arms = {(row["arm"], row.get("change_notice")): row for row in case["results"]}
            if len(arms) != len(case["results"]):
                raise ValueError(f"Duplicate condition in case: {case_id}")
            boundary_digests = {row.get("boundary_sha256") for row in arms.values()}
            if boundary_digests - {None}:
                if len(boundary_digests) != 1:
                    raise ValueError(f"Different shared checkpoints for case: {case_id}")
                shared_boundaries[case_id] = next(iter(boundary_digests))
            for (arm, notice), row in arms.items():
                by_arm[f"{arm}/{notice}" if notice else arm].append(row)
                if notice == "silent":
                    other = arms[arm, "announced"]
                    first = row["first_decision_prompt"] or {}
                    second = other["first_decision_prompt"] or {}
                    notice_pairs.append({
                        "case": case_id, "arm": arm,
                        "same_first_prompt_except_notice": (
                            first.get("notice_present") is False and second.get("notice_present") is True
                            and first["without_notice_sha256"] == second["without_notice_sha256"]
                        ),
                        "silent": {key: row[key] for key in
                                   ("candidate", "passed", "selected_preferred_viable", "continuation_physical_tests")},
                        "announced": {key: other[key] for key in
                                      ("candidate", "passed", "selected_preferred_viable", "continuation_physical_tests")},
                    })
            if ("eager", None) in arms and ("selective", None) in arms:
                eager, selective = arms["eager", None], arms["selective", None]
                pairs.append({
                    "case": case_id,
                    "test_executions_saved": eager["physical_tests"] - selective["physical_tests"],
                    "both_passed": eager["passed"] and selective["passed"],
                    "eager_preferred": eager["selected_preferred_viable"],
                    "selective_preferred": selective["selected_preferred_viable"],
                    "eager_missed_reopening": eager.get("reopening_checks_missed_e1"),
                    "selective_missed_reopening": selective.get("reopening_checks_missed_e1"),
                })
            if ("guarded-eager", None) in arms and ("guarded", None) in arms:
                eager, selective = arms["guarded-eager", None], arms["guarded", None]
                eager_gate = eager["submission_gate"] or {}
                selective_gate = selective["submission_gate"] or {}
                provenance = case.get("provenance", {})
                defect = (f"{provenance['project']}/{provenance['bug']}"
                          if "project" in provenance and "bug" in provenance else case_id)
                scheduler_pairs.append({
                    "case": case_id, "defect": defect,
                    "same_submission_inputs": bool(eager_gate.get("incoming_sha256"))
                    and eager_gate["incoming_sha256"] == selective_gate.get("incoming_sha256"),
                    "tests_saved": eager["physical_tests"] - selective["physical_tests"],
                    "same_selection": eager["candidate"] == selective["candidate"],
                    "acceptance_delta": int(selective["passed"]) - int(eager["passed"]),
                    "preferred_delta": int(selective["selected_preferred_viable"]) - int(eager["selected_preferred_viable"]),
                })
    summary = {}
    for arm, rows in by_arm.items():
        totals = {key: sum(row[key] for row in rows if row.get(key) is not None) for key in METRICS}
        gates = [row["submission_gate"] for row in rows if row.get("submission_gate")]
        summary[arm] = {"cases": len(rows), "totals": totals,
                        "mean_tests": mean(row["physical_tests"] for row in rows),
                        "total_model_tokens": sum(row["model_input_tokens"] + row["model_output_tokens"] for row in rows),
                        "total_invalid_actions": sum(row["invalid_actions"] for row in rows),
                        "two_boundary_cases": sum([c["phase"] for c in row["compactions"]
                                                   if c["reason"] == "length_replay"] == [1, 2] for row in rows)}
        if gates:
            summary[arm]["submission_gate"] = {
                "requests": len(gates),
                "physical_tests": sum(gate["physical_tests"] for gate in gates),
                "changed_candidates": sum(gate["selected"] not in (None, gate["proposal"]) for gate in gates),
                "preferred_proposals": sum(row["submission_gate"]["proposal"] == row["preferred_viable"]
                                           for row in rows if row.get("submission_gate")),
                "verified_fallbacks": sum(gate["reason"] == "verified_fallback" for gate in gates),
                "blocked_submissions": sum(gate["selected"] is None for gate in gates),
            }
        continuations = [row["continuation_wall_seconds"] for row in rows if "continuation_wall_seconds" in row]
        if continuations:
            summary[arm]["total_continuation_wall_seconds"] = sum(continuations)
    report = {"cases": sorted(seen), "rejected": rejected, "arms": summary, "pairs": pairs}
    if shared_boundaries:
        report["shared_boundaries"] = shared_boundaries
    if notice_pairs:
        report["notice_pairs"] = notice_pairs
    if scheduler_pairs:
        report["scheduler_pairs"] = scheduler_pairs
        defects = defaultdict(list)
        for pair in scheduler_pairs:
            defects[pair["defect"]].append(pair)
        report["scheduler_defect_weighted"] = {
            "defects": len(defects), "cases": len(scheduler_pairs),
            "matched_submission_inputs": sum(pair["same_submission_inputs"] for pair in scheduler_pairs),
            **{f"mean_{key}_per_defect": mean(mean(pair[key] for pair in group) for group in defects.values())
               for key in ("tests_saved", "acceptance_delta", "preferred_delta")},
        }
    return report


def matched_comparisons(baseline: list[Path], followup: list[Path]) -> list[dict]:
    def load(paths):
        rows = {}
        for path in paths:
            for case in json.loads(path.read_text(encoding="utf-8")):
                for row in case.get("results", []):
                    key = case["case"], row["arm"]
                    if key in rows:
                        raise ValueError(f"Repeated case and arm: {key}")
                    rows[key] = row
        return rows

    previous, current = load(baseline), load(followup)
    pairs = []
    for base_arm, new_arm in FOLLOWUP_PAIRS:
        for (case, arm), after in current.items():
            if arm != new_arm:
                continue
            if (case, base_arm) not in previous:
                raise ValueError(f"Missing baseline {base_arm} for {case}")
            before = previous[case, base_arm]
            for key in ("preferred_viable", "reference_flips_e1", "reopening_opportunities_e1"):
                if before[key] != after[key]:
                    raise ValueError(f"Reference differs for {case}: {key}")
            pairs.append({
                "case": case, "baseline": base_arm, "followup": new_arm,
                "baseline_candidate": before["candidate"], "followup_candidate": after["candidate"],
                "baseline_passed": before["passed"], "followup_passed": after["passed"],
                "baseline_preferred": before["selected_preferred_viable"],
                "followup_preferred": after["selected_preferred_viable"],
                "baseline_missed_reopening": before["reopening_checks_missed_e1"],
                "followup_missed_reopening": after["reopening_checks_missed_e1"],
                "physical_tests_delta": after["physical_tests"] - before["physical_tests"],
                "submission_gate": after.get("submission_gate"),
            })
    return pairs


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("summary", nargs="+", type=Path)
    cli.add_argument("--baseline", nargs="+", type=Path, help="Compare follow-up arms with archived matched cases")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    report = summarize(args.summary)
    if args.baseline:
        report["matched_comparisons"] = matched_comparisons(args.baseline, args.summary)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
