#!/usr/bin/env bash
set -euo pipefail
umask 077
root=/mnt/sata1/beihang-cuda_backup
mountpoint -q /mnt/sata1
cd "$root"
export HOME="$root/.conda/pilot-home" TMPDIR="$root/.conda/tmp"
export XDG_CACHE_HOME="$root/.conda/runtime" PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
mkdir -p "$HOME" "$TMPDIR" .pilot-rt results/memtrace-pilot
exec 9>.pilot-rt/daemon.lock
flock -n 9 || { echo 'Wait for the active pilot command to finish.' >&2; exit 1; }

p=.conda/memtrace-pilot/bin/python
test -x "$p"
PYTHONPATH="$root/vendor/MemTrace/src" "$p" -c \
  'from memtrace.build_identity import current_build_identity; current_build_identity(require_receipt=True)'
"$p" -m pip install --no-index --no-build-isolation --force-reinstall --no-deps ./vendor/MemTrace
"$p" -m pip check
"$p" -c 'from pathlib import Path; import json; from memtrace.build_identity import current_build_identity; identity = current_build_identity(require_receipt=True); expected = json.loads(Path("vendor/MemTrace/src/memtrace/_build_receipt.json").read_text()); assert identity.build_id == expected["build_id"]; print(identity.build_id)'
lock=$(mktemp results/memtrace-pilot/platform.lock.XXXXXX)
trap 'rm -f -- "$lock"' EXIT
"$p" -m pip freeze --all > "$lock"
mv -- "$lock" results/memtrace-pilot/platform.lock
