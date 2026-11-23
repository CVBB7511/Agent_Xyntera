#!/usr/bin/env bash
set -euo pipefail
root=/mnt/sata1/beihang-cuda_backup
cd "$root"
mountpoint -q /mnt/sata1
export HOME="$root/.conda/pilot-home" TMPDIR="$root/.conda/tmp" XDG_CACHE_HOME="$HOME/cache"
p=.conda/memtrace-pilot/bin/python
"$p" -m pip install --no-cache-dir --index-url https://pypi.org/simple -c code/memtrace_pilot/constraints.txt ./vendor/pilot-data/SWE-EVO/SWE-bench 'pathspec>=0.12'
"$p" -m pip check
"$p" -m pip freeze --all > results/memtrace-pilot/platform.lock
