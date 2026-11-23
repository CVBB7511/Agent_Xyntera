#!/bin/bash
#SBATCH --job-name=irip-torch-smoke
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${1:-python3}"
# Preserve Slurm's CUDA_VISIBLE_DEVICES; cuda:0 means the allocated GPU.
# sbatch already runs this script on an allocated compute node. For this single
# process, execute directly: nested srun failed with cli_filter/chprt_pty on IRIP.
"$PYTHON_BIN" -u code/torch_smoke.py --device cuda --output results/result.json
