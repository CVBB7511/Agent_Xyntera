#!/bin/bash
#SBATCH --job-name=irip-demo-finalize
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
B=/nas/zhangyongkang_2027
E="$B/envs/irip-demo-v2"
L="$B/runs/torch-finalize"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname
"$E/bin/python" - "$B" <<'PY'
import hashlib,json,os,pathlib,shutil,socket,sys,zipfile
b=pathlib.Path(sys.argv[1]);assert b.resolve()==b
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
assert pathlib.Path(sys.prefix)==b/'envs/irip-demo-v2' and sys.prefix!=sys.base_prefix
assert (b/'runs/wheel-recovery/venv-ready.json').exists()
assert (b/'runs/vendor-download/exit-code.txt').read_text().strip()=='0'
manifest=json.loads((b/'runs/wheel-recovery/manifest.json').read_text())
for item in manifest:
    f=b/'packages/recovered-torch-wheels'/item['name'];assert f.resolve()==f
    with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
curated=b/'packages/extra-verified-wheels';assert not curated.exists();curated.mkdir(mode=0o700)
items=json.loads((b/'runs/torch-finalize/extra-sources.json').read_text())
for it in items:
    src=b/'packages/missing-torch-wheels'/it['name'];dst=curated/it['name']
    assert src.stat().st_size==it['bytes'] and src.resolve()==src and not dst.exists()
    shutil.copyfile(src,dst)
expected={x['name']:x['sha256'] for x in json.loads((b/'runs/vendor-download/manifest.json').read_text())}
extras=[]
for f in list(curated.glob('*.whl'))+list((b/'packages/vendor-torch-wheels').glob('*.whl')):
    assert f.resolve()==f
    with zipfile.ZipFile(f) as z:assert z.testzip() is None
    with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if f.name in expected:assert digest==expected[f.name]
    extras.append({'name':f.name,'bytes':f.stat().st_size,'sha256':digest})
assert shutil.disk_usage(b).free>20*1024**3
with (b/'runs/torch-finalize/extra-manifest.json').open('x') as f:json.dump(extras,f,indent=2)
print('ALL_WHEELS_VERIFIED',len(manifest)+len(extras),flush=True)
PY
mkdir -m 700 "$L/tmp"
export TMPDIR="$L/tmp"
"$E/bin/python" -m pip install --isolated --no-cache-dir --no-index --only-binary=:all: \
    --find-links="$B/packages/recovered-torch-wheels" \
    --find-links="$B/packages/extra-verified-wheels" \
    --find-links="$B/packages/vendor-torch-wheels" \
    --find-links="$B/packages/pip-bootstrap" \
    --report "$L/pip-install-report.json" torch==2.9.1
"$E/bin/python" -m pip check
"$E/bin/python" -m pip freeze > "$L/requirements.txt"
"$E/bin/python" -u "$B/projects/demo/release-v1/code/torch_smoke.py" \
    --device cpu --output "$L/cpu-smoke.json"
"$E/bin/python" - "$L" <<'PY'
import json,pathlib,socket,sys,torch
r={'status':'ENVIRONMENT_READY','python':sys.version,'executable':sys.executable,
   'prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':torch.__version__,
   'torch_cuda_runtime':torch.version.cuda,'hostname':socket.gethostname(),
   'gpu_runtime_test':'NOT_RUN_NO_GPU_ALLOCATION'}
with (pathlib.Path(sys.argv[1])/'environment-ready.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
PY
