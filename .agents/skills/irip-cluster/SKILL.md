---
name: irip-cluster
description: Develop locally and safely deploy, run, and debug this project's code on the shared IRIP Slurm cluster. Apply to IRIP SSH, synchronization, resource requests, and result retrieval.
---

# IRIP local-to-cluster workflow

## Sources and verified identity

Read the local manuals when changing resource or environment procedures:
- `服务器使用说明/IRIP集群使用说明_20260120 (1).pdf`, ver0.5, 2026-01-20, especially sections 2, 3, 6, 7.
- `服务器使用说明/IRIP服务器使用须知_20260120 (1).pdf`, ver0.5, 2026-01-20.
Paths above are relative to the project root, three levels above this skill.
Before repeating a connection, deployment or environment setup, read
[the verified environment status](references/environment-status.md) to avoid
repeating already observed failures or confusing completed checks with a working
PyTorch installation.
For storage placement, read [verified mounts and storage constraints](references/storage-layout.md).
The user subsequently explicitly authorized `/nas/zhangyongkang_2027` for temporary
project/environment storage. Its NAS-mapped UID/GID are 1001:100 instead of the
login user's 10080:10078. The user confirmed NAS is shared-access and explicitly
authorized proceeding despite that mapping. Operate only within this exact
canonical root, without changing NAS permissions or storing credentials there.

Endpoint: `115.25.142.114:22`; user: `zhangyongkang_2027`;
expected manager: `irip-c1-manager`; HOME: `/irip/zhangyongkang_2027`.
Identity and HOME were supplied by the user; recheck before deployment.
Windows runs Codex; remote nodes do not need Codex or OpenAI connectivity.

## Cluster requirements from the manuals

- Manager is for login, files and job management. All computation goes through Slurm; never train, compile frameworks, or run nvidia-smi on Manager.
- Debugging uses `interactive`. Query current availability; do not use down/drain nodes or silently change partitions when resources are unavailable.
- Explicitly request CPU, RAM, GPU count/type and walltime. `--gres` is the correct GPU option (one manual example contains a `--grep` typo).
- HOME is shared across cluster nodes. Do not copy the project separately to compute nodes or SSH directly to them.
- Driver versions may differ by node. Determine compatibility in the allocated job, not by changing shared CUDA or drivers.
- Cluster has no data disaster recovery and no Docker support. Preserve local code and important results; never automatically delete data or old deployments.
- rsync is the recommended upload method. Small personal environments may be used only with appropriate authorization; no system Python/CUDA/GCC modifications.
- Repeated failed SSH attempts can cause IP bans. Stop on authentication failure; do not cycle passwords or keys.

The generic server manual discusses cleanup, killing processes, network changes,
and shell startup edits. Those are not authorized by this project: user safety
constraints take precedence. Do not change DNS/proxies, use tunnels, kill jobs or
processes, or run recursive deletion/permission changes without specific approval.

## Deployment and execution boundaries

- Edit code locally only. Inspect actual file lists before excluding paths; a directory named `data` may contain source code.
- Upload only a reviewed allowlist. Exclude credentials, Git history, environments, caches, datasets, weights, checkpoints, logs and results. PDFs are not needed for the smoke test.
- Confirm the canonical remote target is inside the verified user's HOME, not a symlink or another user's directory. Prefer a fresh version directory; never assume `try` is suitable.
- Explicit exception authorized by the user: `/nas/zhangyongkang_2027` and its canonical descendants are approved for this project's temporary code, environments and results. Its shared-access NAS mapping is accepted; do not alter NAS ownership/permissions or store credentials there.
- First upload: list paths and byte sizes, check the target and conflicts, execute dry-run, then obtain the user's confirmation before transfer. Never use rsync `--delete`, `--inplace`, or overwrite unchecked remote files.
- Keep runs/logs outside deployed source files. Preserve Slurm's CUDA_VISIBLE_DEVICES; do not set physical GPU IDs.
- Default synchronization scripts to dry-run, with explicit Apply for transfer. Do not submit tasks as a side effect of synchronization.
- Resource submission needs task-specific authorization. Latest completed installation used 4 CPUs, 16 GiB, a 45-minute limit, interactive partition and no GPU. The user authorized installation and additional CPU/RAM; that is not standing permission for training, GPU submissions or unlimited retries. Verify current state before submitting anything again.
- Record source hashes, exact submission command, Job ID and final exit status. A successful sbatch response is not a successful experiment. Query only necessary job state at moderate intervals; retrieve bounded logs and metrics into a separate local results directory.
- Use existing environments first. If PyTorch is unavailable, report the missing dependency and prepare an isolated installation proposal; do not install automatically.
- Observed 2026-11-19: batch job 66556 obtained an allocation but nested `srun` failed with missing `cli_filter/chprt_pty`. For a single-process batch job, directly execute Python or nvidia-smi within the sbatch script on the allocated node. Do not disable CLI filters or modify shared Slurm plugin configuration. Multi-process launch problems should go to the administrator.
- Observed job 66558: Manager has pip 22.0.2 at `/usr/lib/python3/dist-packages`, but the compute-node interpreter could not import it. Shared HOME does not imply shared system packages. Avoid depending on Manager's pip or ensurepip; an approved isolated installation may use official HTTPS `get-pip.py` within a new venv. Preserve incomplete environments instead of deleting or overwriting them.
- Observed job 66559: HTTPS connection to `bootstrap.pypa.io` timed out on compute-2. This does not establish that all external endpoints are blocked. Do not repeat the same download automatically or change DNS, proxies, tunnels, or gateway settings. Consider an administrator-provided environment or an explicitly approved, checksum-verified offline wheel transfer.
- Passwords and keys must not appear in files, logs, command-line arguments or skills. Prefer temporary process memory for the currently authorized password; authenticate against existing trusted host keys and never disable host-key verification.

## Smoke test

`code/torch_smoke.py` trains a 4-input linear model for 50 steps on 64 synthetic
samples. CPU mode is for local validation. CUDA mode requires a Slurm Job ID,
exactly one visible GPU, and a final MSE below 0.01. It fails rather than silently
falling back to CPU. `code/torch_smoke.sbatch` requests the resource cap above.
Submit from the deployment root with existing `results/` and explicit log paths;
the optional first argument selects an existing Python executable.
