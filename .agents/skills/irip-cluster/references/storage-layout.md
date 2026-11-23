# Storage observations and proposed layout (2026-11-19)

## Latest authorization: NAS mapping accepted, installation completed

The user subsequently confirmed NAS is intentionally shared-access and explicitly
authorized proceeding in `/nas/zhangyongkang_2027`. That resolves the earlier
ownership-check pause below for this exact root. Miniforge and the v2 demo venv
are now installed there; compute-2 NAS visibility was verified using the shared
sentinel. Current details are in `environment-status.md`. The older ownership
observation is retained as history, not a requirement to ask for approval again.

## Earlier ownership check (resolved by user's explicit instruction)

The user explicitly requested creation of `/nas/zhangyongkang_2027`, temporary
project storage there, and resource allocation for installation. This supersedes
the earlier absence of NAS path authorization for this exact path only.

The directory was absent and was created with mode 0700. Immediately afterward:
- Login UID/GID: 10080:10078.
- Directory owner UID/GID: 1001:100; canonical path is unchanged, no symlink.
- The login account can read/write/search the directory despite this mismatch.
- Directory contents are empty. No code, installer, environment or logs were
  written there, and no new installation job was submitted. User queue was empty.

This suggests NAS-side identity mapping or access controls, but the specific
mapping configuration and isolation between laboratory users are NOT verified.
Mode 0700 alone does not prove per-user isolation under an unknown mapping.
The user's original "stop on abnormal target" instruction applies. Obtain
administrator confirmation of UID/GID mapping and private-directory isolation
before further writes. Do not run chown/chmod workarounds, modify NFS/export
configuration, remove the empty directory, or infer installation permission
includes permission to alter server identity/security settings.

NAS mount inspected on Manager: NFSv4.0, `sec=sys`, writable; no `noexec` flag
was shown. Compute-node NAS visibility is still unverified.

## Verified on Manager

`findmnt` and `df` showed all of these as NFSv4 mounts on Manager:

| Mount | Source | Approximate available space at inspection |
| --- | --- | --- |
| /irip | 10.212.253.51:/irip | 14 GiB |
| /irip_16t_0 | 10.212.253.51:/irip_16t_0 | 17 GiB |
| /data | 10.212.253.59:/IRIP-C1-Storage | 27 GiB |
| /nas | 10.134.171.152:/Public/user | 25 TiB |

These values are time-sensitive filesystem availability, not a personal quota.
Do not infer storage performance or permission from available space. Manager's
root disk is separate and must not become a project/data storage workaround.

Compute-2 verified a sentinel file created in the user's HOME by Manager,
confirming that `/irip/zhangyongkang_2027` is shared between those nodes. Mounts
other than HOME have not been verified on compute nodes by this task.

The user explicitly said they do NOT have an administrator-assigned NAS personal
directory. `/nas/zhangyongkang_2027` did not exist at inspection. Do not create a
NAS directory or assume that a world-writable root grants a storage allocation.
First obtain the assigned personal path, owner/group/quota, intended use and
compute-node visibility from the administrator/user. Do not contact anyone on
the user's behalf without explicit permission.

## Placement decision

- Keep authoritative code on local Windows.
- Keep only small deployment code, Slurm scripts and necessary metadata in
  `/irip/zhangyongkang_2027`, within the user's existing HOME-only sync policy.
- After an approved NAS personal path is available and compute-node visibility
  verified, put Miniforge, project environments, package downloads, datasets,
  model weights, checkpoints and substantial experiment output in that personal
  allocation. Do not continue expanding CUDA environments on nearly-full mounts.
- Changing code synchronization to a path outside HOME needs explicit approval;
  never hide an outward-pointing NAS symlink under HOME to bypass that restriction.
- If shared-storage I/O limits training, consider administrator-approved local
  scratch on allocated compute nodes for hot data/job-specific temporary files.
  Check that it exists, is allocated to this user/job, has sufficient space, and
  understand its retention policy first. Never use Manager root or arbitrary
  `/tmp` as an unbounded substitute. Persistent data/results stay on approved
  shared storage; do not automatically delete either persistent or staged data.
- Rebuild environments at the final prefix, or use an explicitly supported and
  approved relocation method. Do not blindly move an installed Conda prefix.

## Performance and resources

Cross-node shared storage is normal. Environment extraction and Python imports
have many small-file/metadata operations; dataset reads and checkpoints use
bandwidth and IOPS. GPU arithmetic does not consult storage for every operation.
Do not promise NAS is faster than other mounts without measurements. No bulk
I/O benchmark or dataset copy was authorized by the storage discussion.

The user allows more resources for installation. A future single CPU-only
installation job may reasonably start with 4 CPUs/8 GiB and a suitable bounded
walltime after storage is settled; this is a proposal, not a GPU training budget.
More CPU/RAM does not guarantee faster NFS operations. Do not increase extraction
parallelism aggressively on a congested shared filesystem.

Job 66561 originally requested 1 CPU, 4 GiB and 15 minutes, without GPU. The user
specifically approved extending it to 45 minutes, but Slurm rejected
`scontrol update JobId=66561 TimeLimit=00:45:00` with permission denied. The limit
remained 15 minutes. Do not retry with sudo or change Slurm configuration.
