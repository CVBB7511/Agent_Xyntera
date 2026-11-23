"""Construct paired decision cases with distinct effective candidate workspaces."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from fixtures import manifest, test_spec, write
from task import Task


SCENARIOS = (
    "prefer-native", "prefer-adapter", "swap-labels", "independent-failure",
    "adapter-regression", "relevant-stable",
)

NATIVE = """from policy import EMPTY_PLACEHOLDER

def indexed(values, start):
    items = list(values)
    if not items:
        return [(start, None)] if EMPTY_PLACEHOLDER else []
    from tqdm.contrib import tenumerate
    return list(tenumerate(items, start=start, tqdm_class=lambda seq, *args, **kwargs: seq))
"""

ADAPTER = """from policy import NORMALIZE_ADAPTER

def indexed(values, start):
    return [
        (start + i, value.strip() if NORMALIZE_ADAPTER else value)
        for i, value in enumerate(values)
    ]
"""

CHECKS = """from numbering import indexed
import json
import sys

def check(name):
    if name == "offset":
        assert indexed(iter(["one", "two"]), 3) == [(3, "one"), (4, "two")]
    elif name == "preserve":
        assert indexed([" x "], 0) == [(0, " x ")]
    elif name == "empty":
        assert indexed([], 2) == []
    elif name == "integration":
        assert indexed((text for text in [" 朱 ", "λ"]), -5) == [(-5, " 朱 "), (-4, "λ")]
        assert indexed(iter([]), -3) == []
    else:
        raise ValueError(name)

try:
    check(sys.argv[1])
except AssertionError:
    status = "assertion_failed"
else:
    status = "passed"
print(json.dumps({"status": status, "detail": sys.argv[1]}))
"""

WRITER_NATIVE = """from pysnooper.pysnooper import get_write_function as writer
"""

WRITER_ADAPTER = """from os import PathLike
from pathlib import Path

def writer(destination):
    if not isinstance(destination, (str, PathLike)):
        return destination.write

    def append(text):
        with Path(destination).open('a') as stream:
            stream.write(text)

    return append
"""

WRITER_CHECKS = """import io
import json
import sys
from pathlib import Path

from trace_output import writer


def check(name):
    if name == 'path':
        writer('trace.log')('line')
        assert Path('trace.log').read_text() == 'line'
    elif name == 'stream':
        buffer = io.StringIO()
        writer(buffer)('stream')
        assert buffer.getvalue() == 'stream'
    elif name == 'integration':
        path = Path('independent.log')
        path.write_text('existing:')
        append = writer(path)
        append('first|')
        append('second')
        assert path.read_text() == 'existing:first|second'
        buffer = io.StringIO()
        writer(buffer)('unchanged')
        assert buffer.getvalue() == 'unchanged'
    else:
        raise ValueError(name)


name = sys.argv[1]
try:
    check(name)
except NameError as exc:
    if exc.name != 'output_path' or name == 'stream':
        raise
    result = 'assertion_failed'
except AssertionError:
    result = 'assertion_failed'
else:
    result = 'passed'
print(json.dumps({'status': result, 'detail': name}))
"""


def workspace_hash(root: Path) -> str:
    files = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    return hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()


def audit_candidates(task: Task) -> dict:
    """Hash the actual file trees delivered by both candidate overlays."""
    with tempfile.TemporaryDirectory(prefix="revalidation-audit-") as temporary:
        root = Path(temporary)
        snapshots = {}
        for epoch in ("E0", "E1"):
            hashes = {}
            for index, candidate in enumerate(task.candidates):
                destination = root / epoch / str(index)
                task.materialize(epoch, candidate, destination)
                hashes[candidate] = workspace_hash(destination)
            snapshots[epoch] = {"hashes": hashes, "distinct": len(set(hashes.values()))}
    return {"case": task.config["id"], "snapshots": snapshots}


def ensure_distinct(path: Path) -> Path:
    audit = audit_candidates(Task(path, sys.executable))
    if any(row["distinct"] != 2 for row in audit["snapshots"].values()):
        raise ValueError(f"Equivalent candidate workspaces in {path}")
    return path


def make_case(parent: Path, upstream: Path, scenario: str, *, reread_after_change: bool = False) -> Path:
    root = parent / f"case-{scenario}"
    native, adapter = ("B", "A") if scenario == "swap-labels" else ("A", "B")
    prefer_adapter = scenario == "prefer-adapter"
    fallback_needs_check = scenario == "independent-blocker-fallback"
    blocked = scenario == "independent-failure" or fallback_needs_check
    adapter_regresses = scenario == "adapter-regression"
    native_recovers = scenario not in {"relevant-stable", "no-change"}
    root.mkdir(parents=True, exist_ok=False)
    for epoch in ("E0", "E1"):
        keep_patch = epoch == "E1" and native_recovers
        shutil.copytree(upstream / ("E1" if keep_patch else "E0") / "tqdm", root / epoch / "tqdm")
        if epoch == "E1" and scenario == "relevant-stable":
            changed = root / epoch / "tqdm" / "contrib" / "__init__.py"
            changed.write_bytes(changed.read_bytes() + b"\n# Packaging revision.\n")
        write(root / epoch, "_v00_check.py", CHECKS)
        write(root / epoch, "policy.py", f"EMPTY_PLACEHOLDER = {blocked}\n"
              f"NORMALIZE_ADAPTER = {epoch == 'E1' and adapter_regresses}\n")
        if fallback_needs_check:
            write(root / epoch, "adapter_policy.py", "NORMALIZE_ADAPTER = False\n"
                  + ("# Configuration touched by the upgrade.\n" if epoch == "E1" else ""))
    write(root / native, "numbering.py", NATIVE)
    adapter_code = ADAPTER.replace("from policy import", "from adapter_policy import") if fallback_needs_check else ADAPTER
    write(root / adapter, "numbering.py", adapter_code)
    common = ["numbering.py", "policy.py"]
    adapter_files = ["numbering.py", "adapter_policy.py"] if fallback_needs_check else common
    upstream_files = [
        path.relative_to(root / "E0").as_posix() for path in sorted((root / "E0" / "tqdm").rglob("*.py"))
    ]
    tests = []
    for name in ("offset", "preserve", "empty"):
        spec = test_spec(name, common + upstream_files, 0.15 if name == "offset" else 0.05)
        spec["candidate_dependencies"] = {
            native: ["_v00_check.py", *common, *(upstream_files if name != "empty" else [])],
            adapter: ["_v00_check.py", *adapter_files],
        }
        tests.append(spec)
    preferred = adapter if prefer_adapter else native
    alternate = native if prefer_adapter else adapter
    goal = (
        "Deliver a numbering frontend that respects the start index, preserves strings, and emits no rows for empty input. "
        + ("Prefer the self-contained adapter because this deployment minimizes package API dependencies."
           if prefer_adapter else "Prefer the upstream-backed implementation to avoid maintaining a second numbering algorithm.")
        + " The other implementation is acceptable if the preferred one cannot meet all requirements."
    )
    native_viable = native_recovers and not blocked
    expected = [
        ["E0", native, "offset", "assertion_failed"],
        ["E0", native, "empty", "assertion_failed" if blocked else "passed"],
        ["E0", adapter, "integration", "passed"],
        ["E1", native, "offset", "passed" if native_recovers else "assertion_failed"],
        ["E1", native, "empty", "assertion_failed" if blocked else "passed"],
        ["E1", native, "integration", "passed" if native_viable else "assertion_failed"],
        ["E1", adapter, "preserve", "assertion_failed" if adapter_regresses else "passed"],
        ["E1", adapter, "integration", "assertion_failed" if adapter_regresses else "passed"],
    ]
    sources = ["tqdm/contrib/__init__.py", "tqdm/std.py", "tqdm/utils.py", "tqdm/tests/tests_tqdm.py",
               "tqdm/cli.py", "tqdm/auto.py", "numbering.py", "policy.py"]
    if fallback_needs_check:
        sources.append("adapter_policy.py")
    initial = [[candidate, test] for candidate in (native, adapter) for test in ("offset", "preserve", "empty")]
    provenance = {"family": "controlled_bugsinpy_variant", "project": "tqdm", "bug": 1, "scenario": scenario,
                  "native": native, "adapter": adapter}
    grade_files = common + upstream_files + (["adapter_policy.py"] if fallback_needs_check else [])
    path = manifest(root, f"decision-tqdm-{scenario}", tests, test_spec("integration", grade_files),
                    sources, "Local adapter", initial, provenance, expected, goal=goal,
                    reread_after_change=reread_after_change)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["candidate_preference"] = [preferred, alternate]
    config["candidates"][native]["description"] = "Use the repository's shipped tqdm numbering behavior."
    config["candidates"][adapter]["description"] = "Use a separate local enumeration implementation."
    write(root, "case.json", json.dumps(config, indent=2) + "\n")
    return ensure_distinct(path)


def make_writer_case(parent: Path, upstream: Path) -> Path:
    root = parent / "case-pysnooper-native"
    root.mkdir(parents=True, exist_ok=False)
    for epoch in ("E0", "E1"):
        shutil.copytree(upstream / epoch, root / epoch, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        write(root / epoch, "_v00_check.py", WRITER_CHECKS)
    write(root / "A", "trace_output.py", WRITER_NATIVE)
    write(root / "B", "trace_output.py", WRITER_ADAPTER)
    common = ["trace_output.py", "_v00_check.py"]
    package = [path.relative_to(root / "E0").as_posix()
               for path in sorted((root / "E0" / "pysnooper").glob("*.py"))]
    tests = []
    for name in ("path", "stream"):
        test = test_spec(name, common + package)
        test["candidate_dependencies"] = {"A": common + package, "B": common}
        tests.append(test)
    original = json.loads((upstream / "case.json").read_text(encoding="utf-8"))
    original_sources = original["prefix_sources"]
    sources = [*original_sources, "trace_output.py"]
    initial = [[candidate, name] for candidate in ("A", "B") for name in ("path", "stream")]
    expected = [
        [epoch, candidate, name, "assertion_failed" if epoch == "E0" and candidate == "A" and name != "stream" else "passed"]
        for epoch in ("E0", "E1") for candidate in ("A", "B") for name in ("path", "stream", "integration")
    ]
    provenance = {**original["provenance"], "family": "controlled_bugsinpy_variant", "scenario": "prefer-native"}
    goal = ("Provide a trace writer that appends to existing files and supports writable streams. "
            "Prefer the built-in PySnooper writer when viable to avoid maintaining separate append code. "
            "A self-contained adapter is acceptable if the built-in writer cannot satisfy these requirements.")
    path = manifest(root, "decision-pysnooper-3-prefer-native", tests, test_spec("integration", common + package),
                    sources, "Use a self-contained append and stream adapter.", initial, provenance, expected,
                    goal=goal, reread_after_change=True)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["candidates"]["A"]["description"] = "Use the PySnooper built-in writer."
    config["candidate_preference"] = ["A", "B"]
    write(root, "case.json", json.dumps(config, indent=2) + "\n")
    return ensure_distinct(path)


TQDM_REGRESSIONS = {
    3: {
        "native": """from io import StringIO
from tqdm import tqdm

def adapt(values):
    with tqdm(values, file=StringIO()) as progress:
        return bool(progress)
""",
        "adapter": """def adapt(values):
    return bool(values)
""",
        "goal": "Report whether a progress input is truthy, even when a generator has no known length.",
        "checks": {
            "generator": "assert adapt((number for number in range(2))) is True",
            "empty": "assert adapt([]) is False",
            "sized": "assert adapt([1]) is True",
            "integration": "assert adapt(iter((3, 5))) is True\nassert adapt(()) is False\nassert adapt((7, 11)) is True",
        },
        "failing": "generator",
    },
    9: {
        "native": "from tqdm._tqdm import format_sizeof as adapt\n",
        "adapter": """def adapt(value, suffix=''):
    for unit in ('', 'K', 'M', 'G', 'T', 'P', 'E', 'Z'):
        if abs(value) < 999.95:
            digits = 2 if abs(value) < 9.995 else 1 if abs(value) < 99.95 else 0
            return f'{value:.{digits}f}{unit}{suffix}'
        value /= 1000
    return f'{value:.1f}Y{suffix}'
""",
        "goal": "Format progress quantities with SI suffixes, promoting the value at a rounding boundary.",
        "checks": {
            "rounding": "assert adapt(999.99).strip() == '1.00K'",
            "sized": "assert adapt(10).strip() == '10.0'",
            "suffix": "assert adapt(12, 'B').strip() == '12.0B'",
            "integration": "assert adapt(999999).strip() == '1.00M'\nassert adapt(20, 'B').strip() == '20.0B'",
        },
        "failing": "rounding",
    },
}


def make_progress_case(parent: Path, upstream: Path, bug: int) -> Path:
    root = parent / f"case-tqdm-{bug}-native"
    source = upstream / f"case-tqdm-{bug}"
    spec = TQDM_REGRESSIONS[bug]
    original = json.loads((source / "case.json").read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=False)
    script = "import json\nimport sys\nfrom progress_behavior import adapt\n\ndef check(name):\n"
    for name, body in spec["checks"].items():
        script += f"    if name == {name!r}:\n" + "".join(f"        {line}\n" for line in body.splitlines()) + "        return\n"
    script += "    raise ValueError(name)\n"
    script += """
name = sys.argv[1]
try:
    check(name)
except AssertionError:
    status = 'assertion_failed'
except TypeError as exc:
    if name not in {'generator', 'integration'} or "'NoneType' object cannot be interpreted as an integer" not in str(exc):
        raise
    status = 'assertion_failed'
else:
    status = 'passed'
print(json.dumps({'status': status, 'detail': name}))
"""
    for epoch in ("E0", "E1"):
        shutil.copytree(source / epoch, root / epoch, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        write(root / epoch, "_v00_check.py", script)
    write(root / "A", "progress_behavior.py", spec["native"])
    write(root / "B", "progress_behavior.py", spec["adapter"])
    package = [path.relative_to(root / "E0").as_posix() for path in sorted((root / "E0" / "tqdm").rglob("*.py"))]
    common = ["progress_behavior.py", "_v00_check.py"]
    names = [name for name in spec["checks"] if name != "integration"]
    checks = []
    for name in names:
        test = test_spec(name, common + package)
        test["candidate_dependencies"] = {"A": common + package, "B": common}
        checks.append(test)
    expected = [
        [epoch, candidate, name, "assertion_failed" if epoch == "E0" and candidate == "A"
         and name in {spec["failing"], "integration"} else "passed"]
        for epoch in ("E0", "E1") for candidate in ("A", "B") for name in spec["checks"]
    ]
    provenance = {**original["provenance"], "family": "controlled_bugsinpy_variant", "scenario": "prefer-native"}
    goal = spec["goal"] + " Prefer the built-in tqdm behavior when viable. A local implementation is an acceptable fallback."
    path = manifest(root, f"decision-tqdm-{bug}-prefer-native", checks, test_spec("integration", common + package),
                    [*original["prefix_sources"], "progress_behavior.py"], "Use a local implementation.",
                    [[candidate, name] for candidate in ("A", "B") for name in names], provenance, expected,
                    goal=goal, reread_after_change=True)
    return ensure_distinct(path)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--upstream-case", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--scenarios", nargs="+", choices=(*SCENARIOS, "no-change", "independent-blocker-fallback"),
                         default=SCENARIOS)
    prepare.add_argument("--reread-after-change", action="store_true")
    writer = commands.add_parser("prepare-writer")
    writer.add_argument("--upstream-case", type=Path, required=True)
    writer.add_argument("--output", type=Path, required=True)
    progress = commands.add_parser("prepare-tqdm-regressions")
    progress.add_argument("--source-directory", type=Path, required=True)
    progress.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--case", nargs="+", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "prepare":
        for scenario in args.scenarios:
            print(make_case(args.output, args.upstream_case, scenario, reread_after_change=args.reread_after_change))
    elif args.command == "prepare-writer":
        print(make_writer_case(args.output, args.upstream_case))
    elif args.command == "prepare-tqdm-regressions":
        for bug in TQDM_REGRESSIONS:
            print(make_progress_case(args.output, args.source_directory, bug))
    else:
        print(json.dumps([audit_candidates(Task(case, sys.executable)) for case in args.case], indent=2))


if __name__ == "__main__":
    main()
