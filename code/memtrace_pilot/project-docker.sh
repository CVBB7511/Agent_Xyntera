#!/usr/bin/env bash
# Start a private daemon only for this invocation and release it on exit.
set -eu
root=/mnt/sata1/beihang-cuda_backup
mountpoint -q /mnt/sata1
cd "$root"
umask 077
export HOME="$root/.conda/pilot-home" XDG_RUNTIME_DIR="$root/.pilot-rt" TMPDIR="$root/.conda/tmp"
export XDG_DATA_HOME="$root/.pilot-docker" XDG_CONFIG_HOME="$HOME/config"
export DOCKER_CONFIG="$HOME/docker-client" DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_DATA_HOME" "$TMPDIR" "$HOME"
exec 9>"$XDG_RUNTIME_DIR/daemon.lock"
flock -n 9 || { echo "private daemon is already in use" >&2; exit 1; }
proxy_env=()
for key in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
  value=${!key:-}
  if [[ "$value" == http://127.0.0.1:* ]]; then proxy_env+=("$key=${value/127.0.0.1/10.0.2.2}"); fi
done
setsid env "${proxy_env[@]}" DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK=false dockerd-rootless.sh --data-root "$XDG_DATA_HOME/data" --exec-root "$XDG_RUNTIME_DIR/exec" --pidfile "$XDG_RUNTIME_DIR/docker.pid" --host "$DOCKER_HOST" >"$root/results/memtrace-pilot/docker.log" 2>&1 9>&- &
daemon=$!
trap 'kill -- -"$daemon" 2>/dev/null || true; wait "$daemon" 2>/dev/null || true' EXIT
for i in {1..30}; do
  kill -0 "$daemon" 2>/dev/null || { tail -12 "$root/results/memtrace-pilot/docker.log" >&2; exit 1; }
  if docker info --format '{{.DockerRootDir}}' 2>/dev/null | grep -Fxq "$XDG_DATA_HOME/data"; then "$@"; exit $?; fi
  sleep 1
done
echo 'project Docker did not become ready' >&2
exit 1
