#!/bin/bash
#SBATCH --job-name=irip-common-libs
#SBATCH --partition=cluster-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
T="${1:?task directory required}"
R="${2:?source release required}"
E=/nas/zhangyongkang_2027/envs/irip-demo-v2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export HF_HOME="$T/install/hf-cache" MPLCONFIGDIR="$T/install/mpl-cache"
export XDG_CACHE_HOME="$T/install/cache"
"$E/bin/python" - "$T" "$R" <<'PY'
import os,pathlib,socket,sys,json,hashlib,zipfile,shutil
b=pathlib.Path('/nas/zhangyongkang_2027');t=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2])
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
assert t.resolve()==t and t.is_relative_to(b) and r.resolve()==r and r.is_relative_to(b)
assert (t/'download/exit-code.txt').read_text().strip()=='0'
assert pathlib.Path(sys.prefix)==b/'envs/irip-demo-v2'
manifest=[]
for f in sorted((t/'download/wheels').glob('*.whl')):
    assert f.resolve()==f
    with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    with zipfile.ZipFile(f) as z:
        assert z.testzip() is None
        expanded=sum(x.file_size for x in z.infolist())
    manifest.append({'name':f.name,'bytes':f.stat().st_size,'expanded_bytes':expanded,'sha256':digest})
assert manifest and shutil.disk_usage(b).free>sum(x['expanded_bytes'] for x in manifest)+20*1024**3
with (t/'download/manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('NAS_VISIBLE',socket.gethostname(),flush=True)
PY
mkdir -m 700 "$T/install/tmp"
export TMPDIR="$T/install/tmp"
"$E/bin/python" -m pip install --isolated --no-cache-dir --no-compile --no-index \
    --find-links="$T/download/wheels" --only-binary=:all: \
    --constraint "$T/download/baseline.txt" --requirement "$R/config/common-libraries.txt" \
    --report "$T/install/pip-report.json"
"$E/bin/python" -m pip check
"$E/bin/python" - "$T" <<'PY'
import importlib,importlib.metadata,json,pathlib,sys
mods=['numpy','scipy','pandas','matplotlib','sklearn','PIL','tqdm','yaml','requests','psutil','rich','pytest','safetensors','huggingface_hub','transformers']
for m in mods: importlib.import_module(m);print('IMPORT_OK',m,flush=True)
import torch
assert torch.__version__=='2.9.1+cu126' and torch.version.cuda=='12.6'
result={'status':'PASS','imports':mods,'versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'torch':torch.__version__}
with (pathlib.Path(sys.argv[1])/'install/libraries-ready.json').open('x') as f:json.dump(result,f,indent=2)
PY
"$E/bin/python" -m pip freeze > "$T/install/requirements.txt"
