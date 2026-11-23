# Project servers

For BYL SSH, project development, Conda environments, GPU runs, and result
retrieval, read `.agents/skills/byl-server/SKILL.md` first.

For IRIP deployment, Slurm execution, environment work, and result retrieval, read
`.agents/skills/irip-cluster/SKILL.md` first.

Write and edit all project code with Codex in this local checkout. Local files
are the only authoritative code. Sync the relevant local files to the selected
server's project directory before running there. Make fixes in the local checkout
and sync them again before rerunning.

Never store credentials in this project. The servers are shared. Do not infer
ongoing GPU submission or environment installation permission from a previous
smoke test.
