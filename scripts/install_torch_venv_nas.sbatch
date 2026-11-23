#!/bin/bash
#SBATCH --job-name=irip-demo-pytorch
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
BASE=/nas/zhangyongkang_2027
ENV="$BASE/envs/irip-demo-v2"
WHEELS="$BASE/packages/torch-cu126-wheels"
LOGROOT="$BASE/runs/torch-install-v2"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname
BASE_PY="$BASE/tools/miniforge3/bin/python"
"$BASE_PY" - "$BASE" <<'PY'
import hashlib,json,os,pathlib,shutil,socket,sys
b=pathlib.Path(sys.argv[1]); assert b.resolve()==b
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
assert (b/'runs/setup/storage-probe.txt').read_text()=='IRIP_NAS_PERSONAL_DIRECTORY_VERIFIED\n'
assert (b/'runs/torch-download/exit-code.txt').read_text().strip()=='0'
manifest=json.loads((b/'runs/torch-download/wheel-manifest.json').read_text())
for item in manifest:
    f=b/'packages/torch-cu126-wheels'/item['name']
    assert f.resolve()==f and f.stat().st_size==item['bytes']
    with f.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
needed=sum(x['expanded_bytes'] for x in manifest)
assert shutil.disk_usage(b).free>needed+20*1024**3
print('Verified all offline wheel hashes',flush=True)
PY
test ! -e "$ENV"
"$BASE_PY" -m venv --without-pip "$ENV"
"$BASE_PY" -m pip --isolated --python "$ENV/bin/python" install --no-cache-dir --no-index --find-links="$BASE/packages/pip-bootstrap" --only-binary=:all: pip setuptools
"$ENV/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix; print("ISOLATED_VENV", sys.prefix)'
mkdir -m 700 "$LOGROOT/tmp"
export TMPDIR="$LOGROOT/tmp"
"$ENV/bin/python" -m pip install --isolated --no-cache-dir --no-index \
    --find-links="$WHEELS" --only-binary=:all: --report "$LOGROOT/pip-install-report.json" torch==2.9.1
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$LOGROOT/requirements.txt"
"$ENV/bin/python" -u "$BASE/projects/demo/release-v1/code/torch_smoke.py" \
    --device cpu --output "$LOGROOT/cpu-smoke.json"
"$ENV/bin/python" -c 'import torch,sys; print("ENVIRONMENT_READY",sys.executable,torch.__version__,torch.version.cuda)'
