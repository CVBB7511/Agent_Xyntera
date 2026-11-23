#!/bin/bash
#SBATCH --job-name=irip-torch-env
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
set -C
BASE="${1:?Pass the verified, fresh deployment directory}"
ENV_PATH="$BASE/env-v2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python3 - "$BASE" <<'PY'
import os, pathlib, shutil, sys
p = pathlib.Path(sys.argv[1])
h = pathlib.Path('/irip/zhangyongkang_2027')
assert pathlib.Path.home() == h and p.parent == h and p.resolve() == p
assert p.stat().st_uid == os.getuid()
for name in ('env-v2', 'install-tmp-v2', 'get-pip-v2.py'):
    assert not (p / name).exists() and not (p / name).is_symlink()
free = shutil.disk_usage(p).free
print('Free bytes before installation:', free, flush=True)
if free < 12 * 1024**3:
    raise RuntimeError('Less than 12 GiB free; stop instead of risking shared storage')
PY
mkdir -m 700 "$BASE/install-tmp-v2"
export TMPDIR="$BASE/install-tmp-v2"
# Compute-node packages differ from Manager; bootstrap using stdlib and HTTPS.
python3 - "$BASE/get-pip-v2.py" <<'PY'
import hashlib, pathlib, sys, urllib.request
url = 'https://bootstrap.pypa.io/get-pip.py'
with urllib.request.urlopen(url, timeout=15) as response:
    data = response.read(3 * 1024 * 1024 + 1)
if len(data) > 3 * 1024 * 1024:
    raise RuntimeError('Unexpected bootstrap download size')
with pathlib.Path(sys.argv[1]).open('xb') as stream:
    stream.write(data)
print('Official get-pip.py SHA256:', hashlib.sha256(data).hexdigest(), flush=True)
PY
python3 -m venv --without-pip "$ENV_PATH"
"$ENV_PATH/bin/python" "$BASE/get-pip-v2.py" --no-setuptools --no-wheel \
    --isolated --no-cache-dir --disable-pip-version-check --timeout 15 --retries 0 \
    --only-binary=:all: --index-url https://pypi.org/simple pip==25.3
"$ENV_PATH/bin/python" -m pip install \
    --isolated --no-cache-dir --disable-pip-version-check --timeout 15 --retries 0 \
    --only-binary=:all: --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1
"$ENV_PATH/bin/python" -m pip freeze > "$BASE/results/requirements.txt"
"$ENV_PATH/bin/python" -c 'import torch; print("INSTALLED", torch.__version__, "CUDA", torch.version.cuda)'
