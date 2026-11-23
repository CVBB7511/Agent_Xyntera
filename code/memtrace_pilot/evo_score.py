import sys
from common import DATA, OUT, read, result_directory
from evo import grade

if __name__ == "__main__":
    task, arm = sys.argv[1:]
    dest = result_directory(OUT) / "runs" / task / arm
    grade(
        read(DATA / "evo" / task / "row.json"),
        dest / "model.patch",
        dest,
        read(OUT / "inputs" / (task + ".json"))["image"],
    )
