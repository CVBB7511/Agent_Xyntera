"""Exercise the same container setup as a formal trial without starting a model."""

from common import OUT, SK, read, release_cached_ref, write
from evo import release_image
from pilot import scoped_environment
from trial import attempt


def main():
    scoped_environment()
    tasks = [*read(OUT / "inputs/evo-screening.json")["qualified"], SK]
    failures = []
    for task in tasks:
        info = read(OUT / "inputs" / (task + ".json"))
        try:
            attempt(task, "native", info, "", "/testbed", prepare_only=True)
            print(f"{task} workspace ready", flush=True)
        except Exception as exc:
            failures.append(task)
            write(OUT / "workspace-probes" / task / "error.json", {"error": str(exc)})
            print(f"{task} workspace failed: {exc}", flush=True)
        finally:
            if task == SK:
                release_cached_ref(info["image"]["tag"])
                release_cached_ref(info["image"]["ref"])
            else:
                release_image(task)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
