#!/bin/bash
# Manager: bounded download-only file transfer. No installation or compilation.
set -euo pipefail
umask 077
BASE=/nas/zhangyongkang_2027
WHEELS="$BASE/packages/torch-cu126-wheels"
LOGROOT="$BASE/runs/torch-download"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -t 60
ulimit -v 1048576
trap 'status=$?; printf "%s\n" "$status" > "$LOGROOT/exit-code.txt"' EXIT
test ! -e "$WHEELS"
mkdir -m 700 "$WHEELS" "$LOGROOT/tmp"
export TMPDIR="$LOGROOT/tmp"
nice -n 19 python3 -m pip download --isolated --no-cache-dir \
    --disable-pip-version-check --timeout 15 --retries 0 --only-binary=:all: \
    --python-version 3.14 --implementation cp --abi cp314 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
    --platform manylinux2014_x86_64 --platform linux_x86_64 \
    --index-url https://download.pytorch.org/whl/cu126 --dest "$WHEELS" torch==2.9.1
python3 - "$WHEELS" "$LOGROOT" <<'PY'
import hashlib, json, pathlib, sys, zipfile
p=pathlib.Path(sys.argv[1]); manifest=[]
for f in sorted(p.glob('*.whl')):
    assert f.resolve()==f
    digest=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    digest=digest.hexdigest()
    with zipfile.ZipFile(f) as z: expanded=sum(x.file_size for x in z.infolist())
    manifest.append({'name':f.name,'bytes':f.stat().st_size,'expanded_bytes':expanded,'sha256':digest})
assert manifest
with (pathlib.Path(sys.argv[2])/'wheel-manifest.json').open('x') as out:
    json.dump(manifest,out,indent=2)
print('DOWNLOAD_READY',len(manifest),'wheels',flush=True)
PY
