"""Command line interface for the evidence revalidation pilot."""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from itertools import product
from pathlib import Path

from experiment import ARMS, BASE_ARMS, Limits, Run, StopRun, prepare_reference
from fixtures import DEMO_SCENARIOS, make_demo, make_tqdm
from model import LocalModel, ScriptedModel, TokenizerSmoke, check_snapshot
from task import Task


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("prepare-demo", help="Generate a synthetic fixture without dependencies")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--scenario", default="fail-to-pass", choices=DEMO_SCENARIOS)
    tqdm = commands.add_parser("prepare-tqdm", help="Export BugsInPy tqdm #1 from local upstream sources")
    tqdm.add_argument("--repo", type=Path, help="Local upstream clone containing both commits")
    tqdm.add_argument("--buggy-archive", type=Path, help="GitHub source archive for the buggy commit")
    tqdm.add_argument("--fixed-archive", type=Path, help="GitHub source archive for the fixed commit")
    tqdm.add_argument("--output", type=Path, required=True)
    tqdm.add_argument("--scenario", default="fail-to-pass", choices=DEMO_SCENARIOS[:3])
    check = commands.add_parser("check", help="Run all public and grading checks for both snapshots and candidates")
    run = commands.add_parser("run", help="Run the paired replay matrix, loading model weights once")
    preflight = commands.add_parser("preflight", help="Check reference results and replay boundaries on CPU")
    snapshot = commands.add_parser("model-check", help="Verify local weight shard structure without CUDA")
    smoke = commands.add_parser("model-smoke", help="Load the model and test one JSON action")
    for subcommand in (snapshot, smoke):
        subcommand.add_argument("--model-dir", type=Path, required=True)
    smoke.add_argument("--device", default="cuda:0")
    check.add_argument("--case", type=Path, required=True)
    for subcommand in (run, preflight):
        subcommand.add_argument("--case", type=Path, nargs="+", required=True)
        subcommand.add_argument("--output", type=Path, required=True)
        subcommand.add_argument("--tokenizer-dir", type=Path, help="Local tokenizer files with no weight loading")
        subcommand.add_argument("--arms", nargs="+", choices=ARMS, default=list(BASE_ARMS))
        subcommand.add_argument("--shared-boundary", action="store_true",
                                help="Run one common prefix per case and reload its second compaction for every arm")
        subcommand.add_argument("--change-notice", choices=("announced", "silent", "paired"), default="announced",
                                help="In paired mode, add the notice after restoring the same silent checkpoint")
        subcommand.add_argument("--window", type=int, default=8192)
        subcommand.add_argument("--trigger", type=int, default=6144)
        subcommand.add_argument("--reserve", type=int, default=1024)
        subcommand.add_argument("--decision-tokens", type=int, default=192, help="Maximum tokens for a single JSON action")
        subcommand.add_argument("--max-model-tokens", type=int, default=110_000)
        subcommand.add_argument("--max-tests", type=int, default=12)
        subcommand.add_argument("--max-test-seconds", type=float, default=90)
        subcommand.add_argument("--max-turns", type=int, default=8, help="Per phase, including invalid responses")
    for subcommand in (check, run, preflight):
        subcommand.add_argument("--test-python", default=sys.executable)
        subcommand.add_argument("--test-timeout", type=float, default=20)
    backend = run.add_mutually_exclusive_group(required=True)
    backend.add_argument("--scripted", action="store_true", help="Run the synthetic protocol smoke backend")
    backend.add_argument("--model-dir", type=Path, help="Local Hugging Face model snapshot directory")
    run.add_argument("--device", default="cuda:0")
    return cli


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    if args.command == "prepare-demo":
        print(make_demo(args.output, args.scenario))
        return 0
    if args.command == "prepare-tqdm":
        from_git = args.repo and not (args.buggy_archive or args.fixed_archive)
        from_archives = not args.repo and args.buggy_archive and args.fixed_archive
        if not (from_git or from_archives):
            cli.error("Use --repo or supply both source archives")
        print(make_tqdm(args.output, args.scenario, args.repo, args.buggy_archive, args.fixed_archive))
        return 0
    if args.command == "model-check":
        print(json.dumps(check_snapshot(args.model_dir), indent=2))
        return 0
    if args.command == "model-smoke":
        model = LocalModel(args.model_dir, args.device)
        prompt = [{"role": "user", "content": 'Reply with exactly this JSON object: {"action":"continue"}'}]
        output = model.complete(prompt, 96, "decision")
        print(json.dumps({"model": model.metadata, "input_tokens": model.count(prompt),
                          "output_tokens": output.output_tokens, "seconds": output.seconds,
                          "text": output.text, "peak_allocated_gpu_bytes": model.peak_memory()}, indent=2))
        try:
            return int(json.loads(output.text) != {"action": "continue"})
        except json.JSONDecodeError:
            return 1
    if not math.isfinite(args.test_timeout) or args.test_timeout <= 0:
        cli.error("--test-timeout must be positive")
    if args.command == "check":
        task = Task(args.case, args.test_python)
        limits = Limits(8192, 6144, 1024, 1, 1, 1, args.test_timeout, 1)
        initial, reference = prepare_reference(task, limits)
        print(json.dumps({"initial": [o.record() for o in initial],
                          "reference": [o.record() for o in reference]}, indent=2))
        return 0
    limits = Limits(args.window, args.trigger, args.reserve, args.max_model_tokens, args.max_tests,
                    args.max_test_seconds, args.test_timeout, args.max_turns, args.decision_tokens)
    if not 0 < limits.trigger < limits.window - limits.reserve or limits.reserve <= 0:
        cli.error("Require 0 < trigger < window - reserve and a positive reserve")
    if (min(limits.max_tokens, limits.max_turns, limits.decision_tokens) <= 0 or limits.max_tests < 0
            or not math.isfinite(limits.max_test_seconds) or limits.max_test_seconds < 0):
        cli.error("Invalid run budget")
    if len(set(args.arms)) != len(args.arms):
        cli.error("--arms must contain unique conditions")
    if args.change_notice == "paired" and not args.shared_boundary:
        cli.error("--change-notice paired requires --shared-boundary")
    scripted = args.command == "preflight" or args.scripted
    if args.tokenizer_dir and not scripted:
        cli.error("--tokenizer-dir requires --scripted")
    output = args.output.resolve()
    cases = [path.resolve() for path in args.case]
    if len(set(cases)) != len(cases):
        cli.error("Use unique case manifests")
    for case in cases:
        if output.is_relative_to(case.parent) or output in case.parent.parents:
            cli.error("Use an output directory separate from the fixture directory")
    output.mkdir(parents=True, exist_ok=False)
    if args.tokenizer_dir:
        model = TokenizerSmoke(args.tokenizer_dir)
    else:
        model = ScriptedModel() if scripted else LocalModel(args.model_dir, args.device)
    rows = []
    for index, case in enumerate(cases, 1):
        case_output = output
        if len(cases) > 1:
            case_output = output / f"case-{index:03d}"
            case_output.mkdir()
        try:
            rows.append(run_case(case, model, limits, args, case_output, scripted, notice_first=index % 2 == 0))
        except (OSError, ValueError, KeyError) as exc:
            if args.command != "preflight":
                raise
            rows.append({"manifest": str(case), "error": str(exc), "ok": False})
            print(f"{case}: {exc}", file=sys.stderr, flush=True)
        write_json(output / "batch_summary.json", rows)
    return int(any(not row["ok"] for row in rows))


def run_case(case: Path, model, limits: Limits, args, output: Path, scripted: bool, *, notice_first: bool = False) -> dict:
    task = Task(case, args.test_python)
    print(f"Case {task.config['id']}", flush=True)
    (output / "case.json").write_bytes(case.read_bytes())
    initial, reference = prepare_reference(task, limits)
    write_json(output / "reference.json", [o.record() for o in reference])
    metadata = {"case": task.config["id"], "model": model.label, "token_unit": model.token_unit,
                "host_python": sys.version, "platform": platform.platform(), "test_python": task.python,
                "device": None if scripted else args.device, "preparation_checks": len(initial) + len(reference),
                "inference_config": getattr(model, "metadata", None)}
    announced = args.change_notice == "announced"
    paired_notice = args.change_notice == "paired"
    metadata.update({"change_notice": args.change_notice, "shared_boundary": args.shared_boundary})
    boundary_path = output / "boundary.json"
    prefix_seconds = 0.0
    if args.shared_boundary:
        if not scripted and model.device.type == "cuda":
            model.torch.cuda.reset_peak_memory_stats(model.device)
        prefix = Run(task, "cache", model, limits, output / "prefix", initial, announce_change=announced)
        started = time.monotonic()
        try:
            prefix.prepare_boundary()
        except StopRun as exc:
            write_json(output / "metadata.json", metadata)
            print(f"Shared prefix stopped: {exc}", flush=True)
            return {"case": task.config["id"], "manifest": str(case), "error": str(exc), "ok": False}
        prefix_seconds = time.monotonic() - started
        write_json(boundary_path, prefix.save_boundary())
        metadata.update({"prefix_policy": "cache", "shared_prefix_wall_seconds": prefix_seconds,
                         "boundary_sha256": hashlib.sha256(boundary_path.read_bytes()).hexdigest()})
    write_json(output / "metadata.json", metadata)
    results = []
    order = ("announced", "silent") if notice_first else ("silent", "announced")
    destinations = {notice: output / notice for notice in order} if paired_notice else {None: output}
    if paired_notice:
        for folder in destinations.values():
            folder.mkdir()
    for arm, notice in product(args.arms, destinations):
        folder = destinations[notice]
        if not scripted and model.device.type == "cuda":
            model.torch.cuda.reset_peak_memory_stats(model.device)
        state = {}
        if args.shared_boundary:
            saved = boundary_path.read_bytes()
            digest = hashlib.sha256(saved).hexdigest()
            if digest != metadata["boundary_sha256"]:
                raise ValueError("The common checkpoint changed before continuation")
            state = {"boundary": json.loads(saved), "boundary_digest": digest, "prefix_seconds": prefix_seconds}
        if paired_notice:
            state["post_boundary_notice"] = notice == "announced"
        result = Run(task, arm, model, limits, folder / arm, initial, announce_change=announced).execute(reference, **state)
        write_json(folder / arm / "summary.json", result)
        results.append(result)
        name = f"{arm}/{notice}" if paired_notice else arm
        print(f"{name:10s} {result['stop']:26s} passed={result['passed']!s:5s} "
              f"compressions={len(result['compactions'])} tests={result['physical_tests']}", flush=True)
    write_json(output / "summary.json", results)
    ok = all([c["phase"] for c in r["compactions"] if c["reason"] == "length_replay"] == [1, 2]
             and (args.command != "preflight" or r["passed"] and r["declared_unchanged_but_flipped"] == 0)
             for r in results)
    return {"case": task.config["id"], "manifest": str(case), "provenance": task.config.get("provenance", {}),
            "results": results, "ok": ok}


if __name__ == "__main__":
    raise SystemExit(main())
