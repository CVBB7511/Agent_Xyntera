# IRIP demo environment (verified 2026-11-19)

Environment and common-library installation are complete. GPU job 66566 is queued;
the user chose to retain the queue entry and track it later. Last observed state:
PENDING (Priority), with no start estimate. Do not submit a duplicate test job.

| Component | Remote absolute path |
| --- | --- |
| Miniforge | `/nas/zhangyongkang_2027/tools/miniforge3` |
| Isolated demo venv | `/nas/zhangyongkang_2027/envs/irip-demo-v2` |
| Demo source | `/nas/zhangyongkang_2027/projects/demo/release-v1/code/torch_smoke.py` |
| Results and dependency lock | `/nas/zhangyongkang_2027/runs/torch-finalize/` |

Python 3.14.10; PyTorch 2.9.1+cu126; bundled CUDA runtime 12.6.
The demo venv has its own packages and depends on the Miniforge base interpreter.
Use its absolute Python path inside an authorized Slurm job. No shell startup
changes or automatic environment activation are needed.

```bash
PYTHONNOUSERSITE=1 /nas/zhangyongkang_2027/envs/irip-demo-v2/bin/python YOUR_SCRIPT.py
```

Run computation on allocated compute nodes, not Manager. GPU allocation and
execution need task-specific authorization; preserve Slurm CUDA_VISIBLE_DEVICES.

Final job 66564 used 4 CPUs, 16 GiB, no GPU, and exited 0:0. `pip check` passed.
The CPU demo passed 50 SGD steps, reducing MSE from 5.80547 to 1.21648e-6.
Common-library job 66565 completed successfully: all 15 import checks and
`pip check` passed. NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1, Matplotlib,
scikit-learn, Pillow, tqdm, PyYAML, requests, psutil, rich, pytest, safetensors,
Hugging Face Hub and Transformers 5.23.0 are available. The earlier NumPy warning
is resolved. Other AI projects should select their own Python/dependencies
and use separate environments rather than changing this validated demo environment.

Local evidence is in `results/nas/`: `environment-ready.json`, `cpu-smoke.json`,
`requirements.txt`, `pip-install-report.json`, logs and `final-accounting.txt`.
The NEW complete dependency lock and library validation are in
`results/libraries-gpu-20261119121521/install-requirements.txt` and
`install-libraries-ready.json`. GPU submission/queue evidence is alongside them.
The GPU task root is
`/nas/zhangyongkang_2027/runs/libraries-gpu-20261119121521/`;
look for `gpu/validation.json` and `gpu/training.json` after job 66566 finishes.
The queued test is limited to one GPU, 2 CPUs, 8 GiB and 5 minutes.
Older failed directories and installer/wheel archives are preserved. No files
were deleted automatically. For future storage migration, recreate this venv and
its Miniforge interpreter at the final prefixes; do not simply move them.
