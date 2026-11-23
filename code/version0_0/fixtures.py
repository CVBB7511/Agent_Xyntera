"""Build reproducible local fixtures. No downloads are performed here."""

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


DEMO_TEST = """from formatter import labels
import json
import sys
import time

def check(name):
    if name == "offset":
        time.sleep(0.12)
        assert labels(["a", "b"], 3) == [(3, "a"), (4, "b")]
    elif name == "empty":
        assert labels([], 3) == []
    elif name == "preserve":
        assert labels([" x "], 0) == [(0, " x ")]
    elif name == "integration":
        assert labels(iter([" x ", "y"]), 5) == [(5, " x "), (6, "y")]
        assert labels([], 1) == []
    else:
        raise ValueError(name)

try:
    check(sys.argv[1])
except AssertionError:
    print(json.dumps({"status": "assertion_failed", "detail": sys.argv[1]}))
else:
    print(json.dumps({"status": "passed", "detail": sys.argv[1]}))
"""

TQDM_TEST = """from tqdm.contrib import tenumerate
from tqdm.std import tqdm
import json
import sys

class Progress(tqdm):
    def __init__(self, iterable, *args, **kwargs):
        super().__init__(iterable, disable=True)

def check(name):
    if name == "offset":
        actual = list(tenumerate(["a", "b"], start=3, tqdm_class=Progress))
        assert actual == [(3, "a"), (4, "b")], repr(actual)
    elif name == "empty":
        assert list(tenumerate([], start=3, tqdm_class=Progress)) == []
    elif name == "integration":
        actual = list(tenumerate(iter([" x ", "y"]), start=5, tqdm_class=Progress))
        assert actual == [(5, " x "), (6, "y")], repr(actual)
        assert list(tenumerate([], start=0, tqdm_class=Progress)) == []
    else:
        raise ValueError(name)

try:
    check(sys.argv[1])
except AssertionError as exc:
    print(json.dumps({"status": "assertion_failed", "detail": str(exc)}))
else:
    print(json.dumps({"status": "passed", "detail": sys.argv[1]}))
"""

BUGGY = "8cc777fe8401a05d07f2c97e65d15e4460feab88"
FIXED = "c0dcf39b046d1b4ff6de14ac99ad9a1b10487512"
DEMO_SCENARIOS = ("unrelated", "relevant-stable", "fail-to-pass", "pass-to-fail", "independent-blocker")


def write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_spec(name: str, depends: list[str], seconds: float = 1) -> dict:
    return {
        "id": name, "argv": ["{python}", "_v00_check.py", name],
        "dependencies": ["_v00_check.py", *depends], "estimated_seconds": seconds,
    }


def manifest(root: Path, case_id: str, tests: list[dict], grade: dict, sources: list[str],
             notes: str, initial: list[list[str]], provenance: dict, expected: list[list[str]],
             *, goal: str = "Deliver a candidate that preserves enumeration inputs and honors the requested start index.",
             reread_after_change: bool = False) -> Path:
    case = {
        "id": case_id,
        "goal": goal,
        "candidate_preference": ["A", "B"],
        "environment_id": "local-python",
        "snapshots": {"E0": "E0", "E1": "E1"},
        "candidates": {
            "A": {"overlay": "A", "description": "Use current repository behavior without a downstream patch."},
            "B": {"overlay": "B", "description": notes},
        },
        "tests": tests,
        "grading_tests": [grade],
        "initial_checks": initial,
        "prefix_sources": sources,
        "readable_sources": sources,
        "provenance": provenance,
        "expected_results": expected,
        "reread_after_change": reread_after_change,
    }
    path = root / "case.json"
    write(root, path.name, json.dumps(case, indent=2) + "\n")
    return path


def make_demo(root: Path, scenario: str) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    (root / "A").mkdir()
    module = (
        "from compatibility import KEEP_EMPTY\n\n"
        "def labels(items, start):\n"
        "    items = list(items)\n"
        "    if not items:\n"
        "        return [] if KEEP_EMPTY else [(start, None)]\n"
        "    from policy import OFFSET_SUPPORTED\n"
        "    return list(enumerate(items, start if OFFSET_SUPPORTED else 0))\n"
    )
    for epoch in ("E0", "E1"):
        stable = epoch == "E0"
        offset = scenario == "pass-to-fail" or (not stable and scenario in {"fail-to-pass", "independent-blocker"})
        revision = 2 if not stable and scenario == "relevant-stable" else 1
        write(root / epoch, "policy.py", f"OFFSET_SUPPORTED = {offset}\nPOLICY_REVISION = {revision}\n")
        keep_empty = scenario != "independent-blocker"
        write(root / epoch, "compatibility.py", f"KEEP_EMPTY = {keep_empty}\n")
        strip = not stable and scenario == "pass-to-fail"
        write(root / epoch, "display.py", f"STRIP_TEXT = {strip}\n")
        write(root / epoch, "formatter.py", module)
        write(root / epoch, "_v00_check.py", DEMO_TEST)
        write(root / epoch, "README.md", f"# Formatting notes\nEpoch document: {epoch}\n")
        lines = [
            f"def sample_{index}(items):\n    return list(enumerate(items, {index}))\n"
            for index in range(950)
        ]
        write(root / epoch, "examples.py", "\n".join(lines))
    candidate = (
        "from display import STRIP_TEXT\n\n"
        "def labels(items, start):\n"
        "    values = (item.strip() if STRIP_TEXT else item for item in items)\n"
        "    return list(enumerate(values, start))\n"
    )
    write(root / "B", "formatter.py", candidate)
    paths = ["formatter.py", "policy.py", "compatibility.py", "display.py", "README.md", "examples.py"]
    a_deps = ["formatter.py", "policy.py", "compatibility.py"]
    b_deps = ["formatter.py", "display.py"]
    tests = []
    for name, seconds in (("offset", 0.15), ("empty", 0.04), ("preserve", 0.04)):
        spec = test_spec(name, a_deps + b_deps, seconds)
        spec["candidate_dependencies"] = {
            "A": ["_v00_check.py", *(a_deps if name != "empty" else ["formatter.py", "compatibility.py"])],
            "B": ["_v00_check.py", *b_deps],
        }
        tests.append(spec)
    checks = [["A", "offset"], ["A", "empty"], ["B", "offset"], ["B", "preserve"]]
    expected = [
        ["E0", "A", "offset", "passed" if scenario == "pass-to-fail" else "assertion_failed"],
        ["E1", "A", "offset", "passed" if scenario in {
            "pass-to-fail", "fail-to-pass", "independent-blocker"} else "assertion_failed"],
        ["E1", "A", "empty", "assertion_failed" if scenario == "independent-blocker" else "passed"],
        ["E1", "A", "integration", "passed" if scenario in {"fail-to-pass", "pass-to-fail"} else "assertion_failed"],
        ["E0", "B", "preserve", "passed"],
        ["E1", "B", "integration", "assertion_failed" if scenario == "pass-to-fail" else "passed"],
    ]
    return manifest(root, f"demo-{scenario}", tests,
        test_spec("integration", a_deps + b_deps), paths, "Apply a local enumeration patch.", checks,
        {"family": "synthetic", "scenario": scenario}, expected)


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def package_files(package: tarfile.TarFile, prefix: str = "") -> dict[str, bytes]:
    files = {}
    for entry in package.getmembers():
        if not entry.isfile() or not entry.name.startswith(prefix + "tqdm/"):
            continue
        name = entry.name.removeprefix(prefix)
        path = Path(name)
        if not path.is_absolute() and ".." not in path.parts:
            with package.extractfile(entry) as stream:
                files[name] = stream.read()
    return files


def make_tqdm(root: Path, scenario: str, repo: Path | None,
              buggy_archive: Path | None, fixed_archive: Path | None) -> Path:
    contrib = "tqdm/contrib/__init__.py"
    if repo is not None:
        source = git_bytes(repo, "archive", "--format=tar", BUGGY, "tqdm")
        with tarfile.open(fileobj=io.BytesIO(source)) as package:
            buggy = package_files(package)
        fixed = git_bytes(repo, "show", f"{FIXED}:{contrib}")
    else:
        with tarfile.open(buggy_archive, "r:*") as package:
            buggy = package_files(package, f"tqdm-{BUGGY}/")
        with tarfile.open(fixed_archive, "r:*") as package:
            fixed = package_files(package, f"tqdm-{FIXED}/")[contrib]
    if contrib not in buggy or buggy[contrib] == fixed:
        raise ValueError("Supplied sources do not contain different buggy and fixed contrib modules")
    root.mkdir(parents=True, exist_ok=False)
    (root / "A").mkdir()
    for name, content in buggy.items():
        destination = root / "E0" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    write(root / "E0", "_v00_check.py", TQDM_TEST)
    shutil.copytree(root / "E0", root / "E1")
    if scenario == "fail-to-pass":
        (root / "E1" / contrib).write_bytes(fixed)
    elif scenario == "relevant-stable":
        target = root / "E1" / contrib
        target.write_bytes(target.read_bytes() + b"\n# Repository maintenance annotation.\n")
    else:
        write(root / "E1", "README.md", "# Updated user documentation\n")
    destination = root / "B" / contrib
    destination.parent.mkdir(parents=True)
    destination.write_bytes(fixed)
    sources = [
        contrib, "tqdm/std.py", "tqdm/utils.py", "tqdm/auto.py", "tqdm/__init__.py",
        "tqdm/contrib/itertools.py", "tqdm/cli.py", "tqdm/tests/tests_tqdm.py",
    ]
    dependencies = [
        p.relative_to(root / "E0").as_posix()
        for p in sorted((root / "E0" / "tqdm").rglob("*.py"))
    ]
    return manifest(root, f"bugsinpy-tqdm-1-{scenario}", [
        test_spec("offset", dependencies), test_spec("empty", dependencies),
    ], test_spec("integration", dependencies), sources, "Use the candidate patch in the progress wrapper.",
        [["A", "offset"], ["B", "offset"], ["B", "empty"]], {
            "family": "BugsInPy", "project": "tqdm", "bug": 1, "scenario": scenario,
            "buggy_commit": BUGGY, "fixed_commit": FIXED,
            "verifier": "Derived enumeration regression with an isolated progress wrapper",
        }, [["E0", "A", "offset", "assertion_failed"],
            ["E1", "A", "offset", "passed" if scenario == "fail-to-pass" else "assertion_failed"],
            ["E1", "B", "integration", "passed"]])
