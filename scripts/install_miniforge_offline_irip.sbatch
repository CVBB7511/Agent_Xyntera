#!/bin/bash
#SBATCH --job-name=irip-miniforge-offline
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
umask 077
JOBROOT="${1:?Verified shared job directory required}"
ACCOUNT_HOME=/irip/zhangyongkang_2027
PREFIX="$ACCOUNT_HOME/tools/miniforge3"
DEMO_ENV="$ACCOUNT_HOME/envs/irip-demo-v1"
INSTALLER="$ACCOUNT_HOME/packages/installers/Miniforge3-26.9.4-2-Linux-x86_64.sh"
EXPECTED_SHA256=bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONDA_REPODATA_THREADS=1 CONDA_EXTRACT_THREADS=1 CONDA_EXECUTE_THREADS=1
export MAMBA_EXTRACT_THREADS=1 MAMBA_DOWNLOAD_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
hostname
python3 - "$JOBROOT" <<'PY'
import json, os, pathlib, shutil, socket, sys
h = pathlib.Path('/irip/zhangyongkang_2027'); j = pathlib.Path(sys.argv[1])
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
assert pathlib.Path.home() == h and h.resolve() == h and h.stat().st_uid == os.getuid()
assert j.parent == h and j.resolve() == j and j.stat().st_uid == os.getuid()
assert (j/'storage-probe.txt').read_text() == 'IRIP_PERSONAL_SHARED_HOME_VERIFIED\n'
for part in ('tools', 'envs', 'packages'):
    p = h/part
    assert p.resolve() == p and p.is_dir() and p.stat().st_uid == os.getuid()
for p in (h/'tools/miniforge3', h/'envs/irip-demo-v1'):
    assert not p.exists() and not p.is_symlink(), f'Refusing existing target: {p}'
if shutil.disk_usage(h).free < 12 * 1024**3:
    raise RuntimeError('Less than 12 GiB free before installation')
print(json.dumps({'compute_host':socket.gethostname(),'shared_directory_verified':True,
                  'free_bytes':shutil.disk_usage(h).free}), flush=True)
PY
printf '%s  %s\n' "$EXPECTED_SHA256" "$INSTALLER" | sha256sum --check -
mkdir -m 700 "$JOBROOT/tmp"
export TMPDIR="$JOBROOT/tmp"
bash "$INSTALLER" -b -p "$PREFIX"

# Clone the bundled Python offline: no system Python/pip/ensurepip or network.
export CONDARC="$JOBROOT/condarc.yml"
export CONDA_PKGS_DIRS="$PREFIX/pkgs:$ACCOUNT_HOME/packages/conda-cache"
export CONDA_ENVS_PATH="$ACCOUNT_HOME/envs"
cat > "$CONDARC" <<'YAML'
channels: []
offline: true
auto_activate_base: false
notify_outdated_conda: false
ssl_verify: true
report_errors: false
repodata_threads: 1
default_threads: 1
YAML
echo 'Creating isolated demo environment from bundled Python, fully offline'
"$PREFIX/bin/conda" create --yes --offline --prefix "$DEMO_ENV" --clone "$PREFIX" \
    --json > "$JOBROOT/conda-clone.json"
"$DEMO_ENV/bin/python" - "$DEMO_ENV" "$JOBROOT" <<'PY'
import json, pathlib, pip, shutil, sys
p = pathlib.Path(sys.argv[1]); j = pathlib.Path(sys.argv[2])
assert pathlib.Path(sys.executable).resolve().is_relative_to(p)
assert pathlib.Path(pip.__file__).resolve().is_relative_to(p)
assert not any(x.startswith('/usr/lib/python') for x in sys.path)
r = {'status':'BASE_ENV_READY','python':sys.version,'executable':sys.executable,
     'prefix':sys.prefix,'pip':pip.__version__,'free_bytes':shutil.disk_usage(p).free}
with (j/'base-environment.json').open('x') as f: json.dump(r, f, indent=2)
print(json.dumps(r), flush=True)
PY
"$PREFIX/bin/conda" list --prefix "$DEMO_ENV" --explicit > "$JOBROOT/conda-explicit.txt"
