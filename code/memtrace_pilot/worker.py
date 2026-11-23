"""Executed inside the benchmark image, with no network or host project mount."""

import hashlib
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


def main(arm, workspace, budget, context_window):
    import httpx
    import yaml
    import minisweagent

    config_dir = Path(minisweagent.__file__).parent / "config"
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import local
    from minisweagent.models.litellm_response_model import LitellmResponseModel

    os.chdir(workspace)
    # Bridge loopback inside this network-isolated container to the model-only socket.
    client = httpx.Client(
        transport=httpx.HTTPTransport(uds="/model/service.sock"), timeout=1800
    )

    class Relay(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            response = client.post(
                "http://model" + self.path,
                content=self.rfile.read(int(self.headers["Content-Length"])),
                headers={"Content-Type": "application/json"},
            )
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.content)

    server = http.server.HTTPServer(("127.0.0.1", 0), Relay)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    os.environ.update(
        OPENAI_API_BASE=base,
        OPENAI_BASE_URL=base,
        OPENAI_API_KEY="local-only",
        PILOT_LOCAL_KEY="local-only",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    real = local._run

    def observe(command, cwd, env, timeout):
        start = time.monotonic()
        entry = {"time": time.time(), "cwd": cwd, "command": command}
        # Snapshot modified/untracked source before a candidate verification.
        # Runtime inputs outside git still need to be checked on manual review.
        if re.search(r"\b(pytest|unittest|tox|ctest)\b", command):
            try:
                paths = subprocess.check_output(
                    ["git", "ls-files", "-m", "-o", "--exclude-standard", "-z"],
                    cwd=workspace,
                    timeout=5,
                ).split(b"\0")
                files = {}
                for name in sorted(set(filter(None, paths))):
                    path = Path(workspace) / os.fsdecode(name)
                    key = os.fsdecode(name)
                    if path.is_symlink() or (
                        path.exists()
                        and (not path.is_file() or path.stat().st_size > 10_000_000)
                    ):
                        files[key] = "unmeasured"
                    elif path.exists():
                        with path.open("rb") as f:
                            files[key] = hashlib.file_digest(f, "sha256").hexdigest()
                    else:
                        files[key] = "deleted"
                entry["changed_files"] = files
                entry["git_head"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
                ).strip()
                entry["index_diff_sha256"] = hashlib.sha256(
                    subprocess.check_output(
                        ["git", "diff", "--cached", "--binary"], cwd=workspace
                    )
                ).hexdigest()
            except (OSError, subprocess.SubprocessError) as exc:
                entry["snapshot_error"] = str(exc)
        try:
            proc = real(command, cwd, env, timeout)
            entry.update(
                returncode=proc.returncode,
                output_tail=proc.stdout[-1600:],
                sha256=hashlib.sha256(proc.stdout.encode()).hexdigest(),
            )
            return proc
        finally:
            entry["seconds"] = time.monotonic() - start
            with Path("/run-out/actions.jsonl").open("a") as f:
                f.write(json.dumps(entry) + "\n")

    local._run = observe
    task = Path("/input/task.txt").read_text()
    if arm == "native":
        config = yaml.safe_load((config_dir / "mini.yaml").read_text())
        config["agent"].pop("mode", None)
        config["agent"].update(
            step_limit=0,
            cost_limit=0,
            wall_time_limit_seconds=budget,
            output_path=Path("/run-out/trajectory.json"),
        )
        config["model"]["model_kwargs"].update(api_base=base, api_key="local-only")
        model = LitellmResponseModel(
            model_name="openai/Qwen3.5-9B",
            cost_tracking="ignore_errors",
            **config["model"],
        )
        agent = DefaultAgent(
            model,
            local.LocalEnvironment(cwd=workspace, **config["environment"]),
            **config["agent"],
        )
        result = agent.run(task)
        agent.save(Path("/run-out/trajectory.json"))
        Path("/run-out/worker-result.json").write_text(json.dumps(result, default=str))
        print(json.dumps(result, default=str), flush=True)
    else:
        from memtrace.cli import main as memtrace

        os.environ["MEMTRACE_MINI_TOKEN_PREFLIGHT"] = "1"
        config = json.loads(Path("/pilot/profile.json").read_text())
        config["provider"].update(
            id="qwen_local",
            name="Local Qwen",
            base_url=base,
            api_key_env="PILOT_LOCAL_KEY",
            model="Qwen3.5-9B",
            model_context_window=context_window,
        )
        config["context_budget"].update(model_limit=context_window, output_reserve=8192)
        config["run_budget"]["wall_clock_seconds"] = budget
        config["redaction_env_vars"].append("PILOT_LOCAL_KEY")
        profile = Path("/run-out/profile.json")
        profile.write_text(json.dumps(config))
        sys.argv = [
            "memtrace",
            "run",
            "--harness",
            "mini_swe_agent",
            "--config",
            str(profile),
            "--task-file",
            "/input/task.txt",
            "--repository",
            workspace,
            "--run-root",
            "/run-out/memtrace",
            "--model",
            "Qwen3.5-9B",
            "--reasoning-effort",
            "high",
        ]
        return memtrace()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
