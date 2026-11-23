#!/usr/bin/env bash
set -euo pipefail
umask 077
root=/mnt/sata1/beihang-cuda_backup
cd "$root"
mountpoint -q /mnt/sata1
df -h .
export HOME="$root/.conda/pilot-home" TMPDIR="$root/.conda/tmp" CONDA_PKGS_DIRS="$root/.conda/pkgs"
export XDG_CACHE_HOME="$HOME/cache" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
mkdir -p "$HOME" "$TMPDIR" results/memtrace-pilot vendor
prefix="$root/.conda/memtrace-pilot"
if ! test -x "$prefix/bin/python"; then
  /home/beihang/anaconda3/bin/conda create --offline --yes --prefix "$prefix" --override-channels -c conda-forge python=3.11 pip
fi
p="$prefix/bin/python"
"$p" -m pip install --index-url https://pypi.org/simple -c code/memtrace_pilot/constraints.txt 'mini-swe-agent==2.4.6' ./vendor/MemTrace
"$p" -m pip check
"$p" -m pip freeze --all > results/memtrace-pilot/platform.lock
