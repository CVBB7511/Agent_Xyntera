#!/bin/bash
#SBATCH --job-name=irip-gpu-verify
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
T="${1:?task directory required}"
R="${2:?source release required}"
E=/nas/zhangyongkang_2027/envs/irip-demo-v2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -f "$T/install/libraries-ready.json"
# Direct single-process execution inside an sbatch GPU allocation; no nested srun.
"$E/bin/python" -u "$R/code/gpu_validate.py" --output "$T/gpu/validation.json"
"$E/bin/python" -u "$R/code/torch_smoke.py" --device cuda --output "$T/gpu/training.json"
