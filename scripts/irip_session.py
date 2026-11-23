"""Temporary local JSON-line SSH session. Run with existing WSL Python/Paramiko.

The first stdin line contains a password, held only in memory. No credential is
accepted through argv or written to disk. Subsequent requests are explicit local
operator commands; this helper does not submit jobs or synchronize by itself.
"""
import json
import os
import pty
import select
import sys
import time

import paramiko

HOST = "115.25.142.114"
USER = "zhangyongkang_2027"
KNOWN_HOSTS = "/mnt/c/Users/Zhang Yongkang/.ssh/known_hosts"


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def rsync_with_password(argv, password, timeout=120):
    if not argv or argv[0] != "rsync":
        raise ValueError("Only rsync is accepted by this transport")
    pid, descriptor = pty.fork()
    if pid == 0:
        os.execvp(argv[0], argv)
    output = bytearray()
    prompted = False
    started = time.monotonic()
    try:
        while True:
            if time.monotonic() - started > timeout:
                raise TimeoutError("rsync transport timed out; inspect before retrying")
            readable, _, _ = select.select([descriptor], [], [], 0.25)
            if not readable:
                continue
            try:
                chunk = os.read(descriptor, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > 1024 * 1024:
                raise RuntimeError("Transport output exceeds 1 MiB")
            if b"password:" in bytes(output[-512:]).lower():
                if not prompted:
                    os.write(descriptor, password.encode() + b"\n")
                    prompted = True
                elif b"permission denied" in bytes(output[-512:]).lower():
                    raise RuntimeError("Authentication failed; no retry permitted")
        _, status = os.waitpid(pid, 0)
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
        return {"exit_code": exit_code,
                "stdout": output.decode(errors="replace").replace(password, "[REDACTED]")}
    finally:
        os.close(descriptor)


def main():
    password = json.loads(sys.stdin.readline())["password"]
    client = paramiko.SSHClient()
    client.load_host_keys(KNOWN_HOSTS)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(HOST, port=22, username=USER, password=password,
                       allow_agent=False, look_for_keys=False, timeout=10,
                       auth_timeout=15, banner_timeout=15)
        emit({"ready": True, "host": HOST, "user": USER})
        for line in sys.stdin:
            request = json.loads(line)
            try:
                if request["action"] == "close":
                    break
                if request["action"] == "ssh":
                    timeout = min(max(int(request.get("timeout", 45)), 1), 300)
                    _, stdout, stderr = client.exec_command(request["command"], timeout=timeout)
                    output = stdout.read(1024 * 1024 + 1)
                    errors = stderr.read(65537)
                    if len(output) > 1024 * 1024 or len(errors) > 65536:
                        raise RuntimeError("Remote output exceeds the allowed size")
                    result = {"exit_code": stdout.channel.recv_exit_status(),
                              "stdout": output.decode(errors="replace"),
                              "stderr": errors.decode(errors="replace")}
                elif request["action"] == "rsync":
                    result = rsync_with_password(request["argv"], password)
                else:
                    raise ValueError("Unknown action")
                emit({"id": request["id"], **result})
            except Exception as exc:
                emit({"id": request["id"], "error": (type(exc).__name__ + ": " + str(exc)).replace(password, "[REDACTED]")})
    except Exception as exc:
        emit({"ready": False, "error": str(exc).replace(password, "[REDACTED]")})
    finally:
        client.close()
        password = None


if __name__ == "__main__":
    main()
