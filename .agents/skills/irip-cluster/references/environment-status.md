# Verified status: 2026-11-19

## Latest: common libraries complete, GPU test queued

The user explicitly requested common libraries and a real GPU test. Library job
66565 COMPLETED 0:0 on compute-1 (cluster-1, 4 CPUs, 16 GiB, no GPU, elapsed 5:24).
`pip check` and all 15 import checks passed. Existing installed dependencies were
constrained using a baseline freeze; PyTorch remains 2.9.1+cu126/CUDA 12.6.

NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1, Matplotlib 3.11.2, scikit-learn 1.9.1,
Pillow, tqdm, PyYAML, requests, psutil, rich, pytest, safetensors, Hugging Face Hub
2.4.0 and Transformers 5.23.0 are installed in the existing demo v2 venv. The
earlier NumPy warning is resolved. See the complete locked requirements below.

GPU job **66566** is ALREADY SUBMITTED. Last observed state: PENDING, reason
Priority; no start estimate. It requests interactive, 1 GPU, 2 CPUs, 8 GiB,
5 minutes, no requeue, and uses direct Python inside the batch allocation.
The user chose: keep it queued, deliver library results this turn, track later.
Do not duplicate the submission, cancel it, or create a monitoring automation.
Its future GPU execution is explicitly authorized; this does not authorize
additional training or arbitrary GPU jobs. No GPU test had run at handoff.

Resume using `squeue -j 66566`, `sacct -j 66566`, and bounded reads of:
- Task root: `/nas/zhangyongkang_2027/runs/libraries-gpu-20261119121521/`.
- GPU logs: `gpu/gpu-66566.out`, `gpu/gpu-66566.err` beneath that root.
- GPU correctness result: `gpu/validation.json`.
- 50-step CUDA training result: `gpu/training.json`.
- Library receipts: `install/libraries-ready.json`, `install/requirements.txt`.
- Source release: `/nas/zhangyongkang_2027/projects/demo/release-20261119121521/`.

The tests require one Slurm-visible CUDA device, compare GPU matrix multiplication
with CPU, check NumPy roundtrip and convolution backward gradients, then run the
tiny linear-regression demo on CUDA. They download no model/data and fail rather
than falling back to CPU. Do not claim GPU success until final job state/exit and
both result artifacts agree.

Local receipts are under `results/libraries-gpu-20261119121521/`, including
`summary.json`, `accounting.txt`, `install-requirements.txt` and submission records.
The full pip report (~984 KB) is retained on the server, not fetched beyond the
bounded-log limit. `.bashrc`/`.profile` hashes remained unchanged; ~/.condarc was
absent. Credentials were kept out of project files and cleared from the local
controller at handoff.

## Most recent NAS installation request

**COMPLETED successfully. This section supersedes all older failure records below.**

The user confirmed NAS shared-access behavior and authorized proceeding within
`/nas/zhangyongkang_2027` despite UID/GID mapping 1001:100. Do not block again
solely on that already-accepted mapping. Do not change NAS permissions or operate
on other users' paths.

- Working Miniforge: `/nas/zhangyongkang_2027/tools/miniforge3`, version 26.9.4-2.
- Working isolated venv: `/nas/zhangyongkang_2027/envs/irip-demo-v2`.
  Python 3.14.10, PyTorch 2.9.1+cu126, bundled CUDA runtime 12.6.
- Demo source: `/nas/zhangyongkang_2027/projects/demo/release-v1/code/`.
- Final results: `/nas/zhangyongkang_2027/runs/torch-finalize/`.
- Job 66562 installed Miniforge successfully but failed during Conda offline
  cloning: some archives were not retained in the installed package cache.
  Do not reinstall Miniforge or blindly repeat `conda create --clone --offline`.
- Job 66563 recovered 16 complete wheels (~3.07 GB), checked CRC and recorded
  SHA256, created the fresh v2 venv from Miniforge with `--without-pip`, and
  bootstrapped its own pip/setuptools using base pip's supported `--python` option.
- Job 66564 COMPLETED, exit 0:0, compute-2, elapsed 9:47, 4 CPUs/16 GiB/0 GPU.
  Offline installation and `pip check` passed. CPU demo passed 50 steps:
  initial MSE 5.805473804473877, final MSE 0.0000012164811096226913.
- GPU execution was NOT tested or submitted in this installation phase. CUDA
  wheel presence is not proof of a successful GPU computation.
- NumPy was absent at this earlier stage; the latest common-library update above
  installed it successfully.
- Final queue was empty; Manager download processes also ended. `.bashrc` and
  `.profile` hashes were unchanged and `~/.condarc` remained absent.
- Receipts, CPU result, requirements and install report: local `results/nas/`.
  A concise usage guide is at project root `IRIP_ENVIRONMENT.md`.

Downloads remain separate from code synchronization. Manager performed bounded
download-only transfers; installations, archive verification and the CPU demo
ran within Slurm allocations. The original Manager pip download hit its CPU
limit (137), a subsequent mirror attempt reached its walltime (124), and all
complete wheels were preserved without deletion. NVIDIA's official `.com`
package endpoints redirected to `.cn` and completed the remaining CUDA downloads;
the last three files matched PyPI official SHA256 digests. Obtain exact wheel
filenames from metadata; don't guess platform tag suffixes or omit compatible
manylinux2010/2014 tags during cross-version downloads.

All older incomplete prefixes, logs and package archives were preserved. Do not
reuse `envs/irip-demo-v1` (failed clone) or the old HOME installation. The v2 venv
depends on the current Miniforge prefix; future migration must recreate at new
paths or use a supported relocation process, not a blind directory move.

## Newer Miniforge attempt and storage decision

This section supersedes the earlier "no Miniforge" observations where relevant.
The user explicitly authorized installing a personal Miniforge and demo
environment, then allowed more CPU/RAM for installation. No GPU training was
authorized by that installation request.

- Job 66560 attempted download/install on compute-2. It failed before installation
  because compute-2 could not reach the TUNA HTTPS endpoint, although Manager
  could. Do not assume networking is identical across nodes.
- Manager downloaded Miniforge 26.9.4-2 from the TUNA mirror into
  `/irip/zhangyongkang_2027/packages/installers/Miniforge3-26.9.4-2-Linux-x86_64.sh`.
  SHA256 `bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf`
  matched the official GitHub release asset digest. The installer is complete.
- Job 66561 ran the offline installer on compute-2 (1 requested CPU, 2 allocated,
  4 GiB, no GPU). It reached TIMEOUT at 15:09 during unpacking. The user approved
  extending to 45 minutes, but Slurm rejected that update; no privilege escalation
  or scheduler changes were attempted.
- `/irip/zhangyongkang_2027/tools/miniforge3` is an INCOMPLETE prefix. At the last
  check it had conda metadata but no `bin/conda` or `bin/python`. Do not treat it
  as a working installation or overwrite/delete it automatically.
- `/irip/zhangyongkang_2027/envs/irip-demo-v1` was not created. No demo environment
  or PyTorch installation succeeded. The installer bundles Python 3.14, so future
  environments should select the project's Python version deliberately.
- `.bashrc` and `.profile` SHA256 values were unchanged; `~/.condarc` remained
  absent. System packages, shell startup, shared Slurm and network settings were
  not changed. Final queue inspection showed no jobs for this user.
- Installation receipts and logs are in local `results/miniforge/`.

Further expansion on nearly-full `/irip` is deferred. The user has no assigned
NAS directory. Read [storage layout and constraints](storage-layout.md); obtain
an administrator-assigned personal path and verify compute-node access before
choosing new persistent environment/data/cache locations. Do not create arbitrary
directories at `/nas` or relocate a partially installed prefix.

## Completed

- Host identity: `irip-c1-manager`, user `zhangyongkang_2027`, canonical HOME `/irip/zhangyongkang_2027`, UID/home owner 10080.
- Windows has PyTorch 2.7.0+cu128. `python -B code/torch_smoke.py --device cpu` passed: initial MSE 5.805473804473877, final MSE 0.0000012165970701971673.
- WSL Ubuntu-20.04 has rsync 3.1.3 and an existing Paramiko installation. Do not assume Windows Python has Paramiko or WSL Python supports newer `os.waitstatus_to_exitcode`.
- rsync dry-run, approved upload and SHA-256 verification succeeded. Only two smoke-test files were uploaded; papers/manuals remain local.
- Current deployed source: `/irip/zhangyongkang_2027/irip-torch-smoke-20261119181039-v2/code/`.
- `torch_smoke.py`: 2519 bytes; SHA256 `e266036ca8109d6ec5b22dc7a3f858f57bb2657af0a2880c7ef7c75f985fa036`.
- `torch_smoke.sbatch`: 680 bytes; SHA256 `be7b254810ccefb1aa6633b7787c9495551df2fb31f715574c610e402b6b233e`.
- Driver probe 66557 completed on compute-2: NVIDIA A800 80GB PCIe, driver 580.178.04. Requested 1 CPU and 1 GPU; Slurm accounted for 2 CPUs. User explicitly accepted this scheduling granularity.

## Jobs and blockers

| Job | Purpose | Final result |
| --- | --- | --- |
| 66556 | Initial GPU driver probe | FAILED, nested srun missing cli_filter/chprt_pty |
| 66557 | Direct command within sbatch allocation | COMPLETED, exit 0:0 |
| 66558 | Personal venv bootstrap using Manager pip path | FAILED, compute-node interpreter could not import pip |
| 66559 | Personal venv bootstrap using official HTTPS get-pip | FAILED, connection to bootstrap.pypa.io timed out |

No GPU PyTorch test was submitted. Do not report a successful GPU training run.
Default remote Python is 3.10.12 without PyTorch; no working personal PyTorch
environment was found in common locations. No system packages were changed.

The deployment's `env/` is an incomplete venv, not a usable PyTorch environment.
`env-v2/` does not exist: the bootstrap download failed before venv creation.
The original deployment without `-v2`, temporary installation directories and
all logs were preserved. Do not delete or repurpose them without approval.

At the last check, the user's Slurm queue was empty. The user authorized each
attempt individually and required stopping on failure; no remaining submission
or installation permission should be inferred. A future attempt requires a
concrete approved plan, preferably an existing laboratory environment path or
offline packages with an audited manifest, sizes, checksums and disk budget.

## Local artifacts and credential handling

- `results/irip/summary.json`: machine-readable outcome.
- `results/irip/all-jobs.txt`: final Slurm accounting and empty queue check.
- `results/irip/probe-*.{out,err}` and `install-*.{out,err}`: downloaded bounded logs.
- `scripts/install_irip_env.sbatch`: latest attempted installation method; do not rerun blindly.
- `scripts/irip_session.py`: local WSL JSON-line transport using existing Paramiko and password-in-memory. It executes only explicit requests and performs no automatic submissions on startup. Keep host-key verification enabled.
- `results/irip/trusted_known_hosts`: public host keys copied from the existing Windows trust store for OpenSSH's rsync transport; no new host trust was accepted.
- No password, key or token was written into project files. The temporary controller's password variable was cleared after the final retrieval. Passwords must be supplied through transient stdin/process memory, never argv, logs or saved scripts.

All artifact paths in this section are relative to the project root. `results/`
is ignored by Git and excluded from the smoke upload allowlist.
