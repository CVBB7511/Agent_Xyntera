"""Prepare additional BugsInPy cases from pinned upstream source archives."""

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from fixtures import manifest, test_spec, write


TQDM_IMPORTS = "from io import StringIO\nfrom tqdm import tqdm"
TQDM_SANITY = "assert list(tqdm([1, 2], disable=True, file=StringIO())) == [1, 2]"
CASES = [
    ("tqdm", 3, "c2599e3cd6087429f48bae34347ec5d2473c8392", "73962a47026dd980ac0758820efc9c41cbf938e0",
     "tqdm/_tqdm.py", "Preserve the truthiness of wrapped generators with unknown length.",
     TQDM_IMPORTS, "TypeError",
     "with tqdm((x for x in range(2)), file=StringIO()) as bar:\n    assert bool(bar)",
     TQDM_SANITY,
     "with tqdm((x for x in range(4)), disable=False, file=StringIO()) as bar:\n"
     "    assert bool(bar)\n    assert list(bar) == [0, 1, 2, 3]"),
    ("tqdm", 4, "03b347646492131d889871939b40457d29147216", "964dee631d0ed30e2f799b42fc58ba5e73795a08",
     "tqdm/_tqdm.py", "Format scaled counters even when the total is unknown.",
     TQDM_IMPORTS, "TypeError",
     "assert '30it' in tqdm.format_meter(3, None, 3, unit_scale=10)",
     TQDM_SANITY,
     "assert '28it' in tqdm.format_meter(7, None, 5, unit_scale=4)"),
    ("tqdm", 5, "19b08ab34fdbfa0275bc5cb2430436c724c7e759", "4f340697af69b71850aad496387c9c5aa1904136",
     "tqdm/_tqdm.py", "Preserve the total when progress display is disabled.",
     TQDM_IMPORTS, "AttributeError",
     "with tqdm(total=10, disable=True, file=StringIO()) as bar:\n    assert bool(bar)",
     TQDM_SANITY,
     "with tqdm(total=0, disable=True, file=StringIO()) as bar:\n    assert not bool(bar)"),
    ("tqdm", 8, "08b8ad1ff3bd003ef8309faaa0cc108ffa40317d", "cae9d139c6df5614be3bf6e25ccbd600ee3286dc",
     "tqdm/_tqdm.py", "Honor both sides of a custom progress-bar format.",
     TQDM_IMPORTS, "AssertionError",
     "line = tqdm.format_meter(20, 100, 12, ncols=16, ascii=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')\n"
     "assert line.startswith(' 20%|') and line.endswith('|20/100'), line",
     "assert '3it' in tqdm.format_meter(3, None, 1)",
     "line = tqdm.format_meter(25, 100, 15, ncols=18, ascii=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')\n"
     "assert line.startswith(' 25%|') and line.endswith('|25/100'), line"),
    ("tqdm", 9, "9da1d5d116aec7a23d8f6dc22d5e23ecb1c40a7c", "d877c1dfb4739852105f7b967a8fceb869ac5042",
     "tqdm/_tqdm.py", "Use the next SI suffix when rounding crosses the unit threshold.",
     "from tqdm._tqdm import format_sizeof", "AssertionError",
     "assert format_sizeof(999.99).strip() == '1.00K'",
     "assert format_sizeof(10).strip() == '10.0'",
     "assert format_sizeof(999999).strip() == '1.00M'"),
    ("PySnooper", 3, "6e3d797be3fa0a746fb5b1b7c7fea78eb926c208", "15555ed760000b049aff8fecc79d29339c1224c3",
     "pysnooper/pysnooper.py", "Write traces to the requested output file.",
     "import io\nfrom pathlib import Path\nfrom pysnooper.pysnooper import get_write_function", "NameError",
     "get_write_function('trace.log')('trace line')\nassert Path('trace.log').read_text() == 'trace line'",
     "out = io.StringIO()\nget_write_function(out)('stream')\nassert out.getvalue() == 'stream'",
     "path = Path('alternate.log')\nget_write_function(path)('first')\nget_write_function(path)('second')\n"
     "assert path.read_text() == 'firstsecond'"),
]


def read_sources(archive: Path, revision: str) -> dict[str, bytes]:
    sources = {}
    with tarfile.open(archive, "r:*") as package:
        for entry in package:
            path = PurePosixPath(entry.name)
            if not entry.isfile() or len(path.parts) < 2 or not path.parts[0].endswith(f"-{revision}"):
                continue
            relative = PurePosixPath(*path.parts[1:])
            if ".." in relative.parts or relative.suffix.lower() not in {".py", ".md", ".rst", ".txt"}:
                continue
            with package.extractfile(entry) as stream:
                sources[relative.as_posix()] = stream.read()
    return sources


def make_case(parent: Path, archives: Path, case: tuple) -> Path:
    project, bug, buggy_id, fixed_id, target, goal, imports, known_error, regression, sanity, integration = case
    root = parent / f"case-{project.lower()}-{bug}"
    buggy = read_sources(archives / f"{project}-{bug}-buggy.tar.gz", buggy_id)
    fixed = read_sources(archives / f"{project}-{bug}-fixed.tar.gz", fixed_id)
    if buggy[target] == fixed[target]:
        raise ValueError(f"Identical target in case {project}/{bug}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "A").mkdir()
    for name, content in buggy.items():
        path = root / "E0" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # Import failures remain infrastructure errors. Only the documented failure
    # type during the reproducer is accepted as a failing assertion.
    script = f"import sys, json\n{imports}\n\n"
    checks = {"regression": regression, "sanity": sanity, "integration": integration}
    for name, body in checks.items():
        script += f"def {name}():\n" + "".join(f"    {line}\n" for line in body.splitlines()) + "\n"
    script += f"""name = sys.argv[1]
try:
    {{"regression": regression, "sanity": sanity, "integration": integration}}[name]()
except (AssertionError, {known_error}) as exc:
    if name == "sanity" and not isinstance(exc, AssertionError):
        raise
    print(json.dumps({{"status": "assertion_failed", "detail": str(exc)}}))
else:
    print(json.dumps({{"status": "passed", "detail": name}}))
"""
    write(root / "E0", "_v00_check.py", script)
    shutil.copytree(root / "E0", root / "E1")
    (root / "E1" / target).write_bytes(fixed[target])
    patch = root / "B" / target
    patch.parent.mkdir(parents=True)
    patch.write_bytes(fixed[target])
    dependencies = sorted(name for name in buggy if name.startswith(project.lower() + "/") and name.endswith(".py"))
    sources = [target, *sorted(name for name in buggy if name != target)]
    expected = [
        ["E0", "A", "regression", "assertion_failed"], ["E0", "B", "regression", "passed"],
        ["E1", "A", "regression", "passed"], ["E1", "A", "integration", "passed"],
        ["E1", "B", "integration", "passed"],
    ]
    return manifest(root, f"bugsinpy-{project.lower()}-{bug}-fail-to-pass",
                    [test_spec(name, dependencies) for name in ("regression", "sanity")],
                    test_spec("integration", dependencies), sources, "Apply the candidate patch.",
                    [["A", "regression"], ["B", "regression"], ["B", "sanity"]],
                    {"family": "BugsInPy", "project": project, "bug": bug, "scenario": "fail-to-pass",
                     "buggy_commit": buggy_id, "fixed_commit": fixed_id, "verifier": "Derived minimal checks"},
                    expected, goal=goal, reread_after_change=True)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archives", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    for case in CASES:
        print(make_case(args.output, args.archives, case))


if __name__ == "__main__":
    main()
