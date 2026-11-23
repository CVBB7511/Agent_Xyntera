#!/bin/bash
#SBATCH --job-name=irip-demo-prepare
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
P="$B/tools/miniforge3/bin/python"
E="$B/envs/irip-demo-v2"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$P" - "$B" <<'PY'
import concurrent.futures,hashlib,json,os,pathlib,shutil,socket,sys,zipfile
b=pathlib.Path(sys.argv[1]);assert b.resolve()==b
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
items=json.loads((b/'runs/wheel-recovery/sources.json').read_text())
target=b/'packages/recovered-torch-wheels';assert not target.exists();target.mkdir(mode=0o700)
def copy(item):
    src=pathlib.Path(item['source']);dst=target/item['name']
    assert src.resolve()==src and src.is_relative_to(b/'runs/torch-download/tmp')
    assert src.stat().st_size==item['bytes'] and not dst.exists()
    shutil.copyfile(src,dst)
    with zipfile.ZipFile(dst) as z:
        assert z.testzip() is None
    with dst.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'name':dst.name,'bytes':dst.stat().st_size,'expanded_bytes':item['expanded_bytes'],'sha256':digest}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    manifest=list(pool.map(copy,items))
with (b/'runs/wheel-recovery/manifest.json').open('x') as out:json.dump(manifest,out,indent=2)
print('RECOVERED_AND_VERIFIED',len(manifest),flush=True)
PY
test ! -e "$E"
"$P" -m venv --without-pip "$E"
"$P" -m pip --isolated --python "$E/bin/python" install --no-cache-dir --no-index \
    --find-links="$B/packages/pip-bootstrap" --only-binary=:all: pip setuptools
"$E/bin/python" - "$B" <<'PY'
import json,pathlib,pip,sys
b=pathlib.Path(sys.argv[1]); e=b/'envs/irip-demo-v2'
assert sys.prefix!=sys.base_prefix and pathlib.Path(sys.prefix)==e
assert pathlib.Path(pip.__file__).is_relative_to(e)
assert not any(x.endswith('/tools/miniforge3/lib/python3.14/site-packages') for x in sys.path)
with (b/'runs/wheel-recovery/venv-ready.json').open('x') as f:
    json.dump({'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'pip':pip.__version__},f,indent=2)
print('ISOLATED_VENV_READY',sys.prefix,flush=True)
PY
