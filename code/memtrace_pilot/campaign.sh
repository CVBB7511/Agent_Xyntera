#!/usr/bin/env bash
set -euo pipefail
root=/mnt/sata1/beihang-cuda_backup
cd "$root"
mountpoint -q /mnt/sata1
export HOME="$root/.conda/pilot-home" XDG_CACHE_HOME="$root/.conda/runtime" TMPDIR="$root/.conda/tmp"
export PYTHONPATH="$root/code/memtrace_pilot" PYTHONNOUSERSITE=1
export PILOT_CONTEXT_WINDOW=131072
out="$root/results/memtrace-pilot"
umask 077
if [[ "${1:-}" != private ]]; then
  exec 8>"$out/campaign.lock"
  flock -n 8 || { echo 'A pilot campaign is already running.'; exit 1; }
  # A CPU-only control batch may already own the project daemon.
  flock "$root/.pilot-rt/daemon.lock" true
  exec bash code/memtrace_pilot/project-docker.sh bash code/memtrace_pilot/campaign.sh private
fi
p="$root/.conda/v00/bin/python"
judge="$root/.conda/memtrace-pilot/bin/python"
for test in smoke-context smoke-tools; do test -s "$out/$test.json"; done
"$p" -c 'from common import *; from benchdata import prefix; import json; refs=read(OUT/"inputs/evo.json"); [(write(OUT/"inputs"/(r["instance_id"]+".json"),dict(id=r["instance_id"],base=r["base_commit"],image=image_manifest(r["image"]+":latest"),timeout=14400))) for r in refs["candidates"] if not (OUT/"inputs"/(r["instance_id"]+".json")).exists()]'
for bench in DeepSWE SWE-EVO SWE-Milestone; do
  case "$bench" in
    DeepSWE) "$p" code/memtrace_pilot/deepswe.py ;;
    SWE-EVO) "$judge" -c 'import evo; evo.controls()' ;;
    SWE-Milestone) "$judge" -c 'import mile; mile.controls()' ;;
  esac
  "$p" code/memtrace_pilot/pilot.py run --private --benchmark "$bench"
  "$p" code/memtrace_pilot/pilot.py report
  if [[ "$bench" == DeepSWE ]]; then
    "$p" -c 'from common import *; [release_cached_ref(read(OUT/"inputs"/(t+".json"))["image"]["ref"]) for t in DEEP if all((OUT/"runs"/t/a/"grade.json").exists() for a in ("native","memtrace"))]'
  fi
done
