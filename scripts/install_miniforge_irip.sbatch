#!/bin/bash
#SBATCH --job-name=irip-miniforge
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
umask 077
JOBROOT="${1:?Verified shared job directory required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONDA_REPODATA_THREADS=1 CONDA_EXTRACT_THREADS=1 CONDA_EXECUTE_THREADS=1
export MAMBA_EXTRACT_THREADS=1 MAMBA_DOWNLOAD_THREADS=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ACCOUNT_HOME=/irip/zhangyongkang_2027
PREFIX="$ACCOUNT_HOME/tools/miniforge3"
DEMO_ENV="$ACCOUNT_HOME/envs/irip-demo-py310-v1"
CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
INSTALLER_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/Release%2026.9.4-2/Miniforge3-26.9.4-2-Linux-x86_64.sh
EXPECTED_SHA256=bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf

hostname
python3 - "$JOBROOT" <<'PY'
import json, os, pathlib, shutil, socket, sys
h = pathlib.Path('/irip/zhangyongkang_2027')
j = pathlib.Path(sys.argv[1])
assert os.environ.get('SLURM_JOB_ID') and '-compute-' in socket.gethostname()
assert pathlib.Path.home() == h and h.resolve() == h and h.stat().st_uid == os.getuid()
assert j.parent == h and j.resolve() == j and j.stat().st_uid == os.getuid()
assert (j/'storage-probe.txt').read_text() == 'IRIP_PERSONAL_SHARED_HOME_VERIFIED\n'
for part in ('tools', 'envs', 'packages'):
    p = h/part
    assert not p.is_symlink()
    if not p.exists(): p.mkdir(mode=0o700)
    assert p.is_dir() and p.stat().st_uid == os.getuid() and p.resolve() == p
for p in (h/'tools/miniforge3', h/'envs/irip-demo-py310-v1'):
    assert not p.exists() and not p.is_symlink(), f'Refusing existing target: {p}'
if shutil.disk_usage(h).free < 12 * 1024**3:
    raise RuntimeError('Less than 12 GiB free before bootstrap; stopping')
print(json.dumps({'compute_host':socket.gethostname(),'shared_directory_verified':True,
                  'free_bytes':shutil.disk_usage(h).free}), flush=True)
PY

mkdir -m 700 "$JOBROOT/tmp"
export TMPDIR="$JOBROOT/tmp"
INSTALLER="$JOBROOT/Miniforge3-26.9.4-2-Linux-x86_64.sh"
test ! -e "$INSTALLER"
curl --fail --location --show-error --silent --proto '=https' --proto-redir '=https' \
    --connect-timeout 8 --max-time 180 --retry 0 --limit-rate 8M \
    --output "$INSTALLER" "$INSTALLER_URL"
printf '%s  %s\n' "$EXPECTED_SHA256" "$INSTALLER" | sha256sum --check -
bash "$INSTALLER" -b -p "$PREFIX"

# Isolate configuration and caches without touching ~/.condarc or shell startup.
export CONDARC="$JOBROOT/condarc.yml"
export CONDA_PKGS_DIRS="$ACCOUNT_HOME/packages/conda-cache"
export CONDA_ENVS_PATH="$ACCOUNT_HOME/envs"
cat > "$CONDARC" <<YAML
channels:
  - $CHANNEL
channel_priority: strict
auto_activate_base: false
notify_outdated_conda: false
ssl_verify: true
report_errors: false
remote_connect_timeout_secs: 8
remote_read_timeout_secs: 45
remote_max_retries: 0
repodata_threads: 1
default_threads: 1
YAML
python3 - "$ACCOUNT_HOME" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 10 * 1024**3:
    raise RuntimeError('Less than 10 GiB free after Miniforge; stop before creating demo env')
PY
echo 'Creating isolated demo Python environment'
"$PREFIX/bin/conda" create --yes --prefix "$DEMO_ENV" --override-channels \
    --channel "$CHANNEL" --solver libmamba --repodata-fn current_repodata.json \
    python=3.10 pip --json > "$JOBROOT/conda-create.json"
"$DEMO_ENV/bin/python" - "$DEMO_ENV" "$JOBROOT" <<'PY'
import json, pathlib, pip, shutil, sys
p = pathlib.Path(sys.argv[1])
assert pathlib.Path(sys.executable).resolve().is_relative_to(p)
assert pathlib.Path(pip.__file__).resolve().is_relative_to(p)
result = {'status':'BASE_ENV_READY','python':sys.version,'executable':sys.executable,
          'prefix':sys.prefix,'pip':pip.__version__,'free_bytes':shutil.disk_usage(p).free}
with (pathlib.Path(sys.argv[2])/'base-environment.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
PY
"$PREFIX/bin/conda" list --prefix "$DEMO_ENV" --explicit > "$JOBROOT/conda-explicit.txt"
