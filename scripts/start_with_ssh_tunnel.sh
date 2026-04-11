#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

SSH_HOST="${FACE_SERVICE_SSH_HOST:-}"
SSH_USER="${FACE_SERVICE_SSH_USER:-}"
SSH_PORT="${FACE_SERVICE_SSH_PORT:-22}"
SSH_PASSWORD="${FACE_SERVICE_SSH_PASSWORD:-}"

LOCAL_HOST="${FACE_SERVICE_SSH_LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${FACE_SERVICE_SSH_LOCAL_PORT:-18000}"
REMOTE_HOST="${FACE_SERVICE_SSH_REMOTE_HOST:-127.0.0.1}"
REMOTE_PORT="${FACE_SERVICE_SSH_REMOTE_PORT:-8000}"

BASE_URL="${FACE_SERVICE_BASE_URL:-http://${LOCAL_HOST}:${LOCAL_PORT}}"
CONTROL_SOCKET="/tmp/cd_face_${USER:-user}_$$.sock"
TUNNEL_STARTED=0

cleanup() {
  if [ "$TUNNEL_STARTED" -eq 1 ] && [ -S "$CONTROL_SOCKET" ]; then
    ssh \
      -S "$CONTROL_SOCKET" \
      -O exit \
      -p "$SSH_PORT" \
      "${SSH_USER}@${SSH_HOST}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

port_is_open() {
  nc -z "$LOCAL_HOST" "$LOCAL_PORT" >/dev/null 2>&1
}

open_tunnel_with_expect() {
  SSH_EXPECT_PASSWORD="$SSH_PASSWORD" \
  SSH_EXPECT_HOST="$SSH_HOST" \
  SSH_EXPECT_USER="$SSH_USER" \
  SSH_EXPECT_PORT="$SSH_PORT" \
  SSH_EXPECT_LOCAL_HOST="$LOCAL_HOST" \
  SSH_EXPECT_LOCAL_PORT="$LOCAL_PORT" \
  SSH_EXPECT_REMOTE_HOST="$REMOTE_HOST" \
  SSH_EXPECT_REMOTE_PORT="$REMOTE_PORT" \
  SSH_EXPECT_CONTROL_SOCKET="$CONTROL_SOCKET" \
  expect <<'EOF'
set timeout 20
set password $env(SSH_EXPECT_PASSWORD)
set host $env(SSH_EXPECT_HOST)
set user $env(SSH_EXPECT_USER)
set port $env(SSH_EXPECT_PORT)
set local_host $env(SSH_EXPECT_LOCAL_HOST)
set local_port $env(SSH_EXPECT_LOCAL_PORT)
set remote_host $env(SSH_EXPECT_REMOTE_HOST)
set remote_port $env(SSH_EXPECT_REMOTE_PORT)
set control_socket $env(SSH_EXPECT_CONTROL_SOCKET)

spawn ssh \
  -f \
  -M \
  -S $control_socket \
  -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L ${local_host}:${local_port}:${remote_host}:${remote_port} \
  -p $port \
  ${user}@${host} \
  -N

expect {
  -re "(?i)password:" {
    send -- "$password\r"
    exp_continue
  }
  -re "(?i)permission denied" {
    exit 1
  }
  eof {
    catch wait result
    set exit_status [lindex $result 3]
    exit $exit_status
  }
  timeout {
    exit 1
  }
}
EOF
}

ensure_tunnel() {
  if [ -z "$SSH_HOST" ] || [ -z "$SSH_USER" ]; then
    printf '%s\n' "请先设置 FACE_SERVICE_SSH_HOST 和 FACE_SERVICE_SSH_USER。" >&2
    exit 1
  fi

  if port_is_open; then
    printf '%s\n' "检测到本地 ${LOCAL_HOST}:${LOCAL_PORT} 已有监听，直接复用。"
    return 0
  fi

  printf '%s\n' "正在建立 SSH 隧道 ${LOCAL_HOST}:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
  if [ -n "$SSH_PASSWORD" ] && command -v expect >/dev/null 2>&1; then
    printf '%s\n' "检测到 FACE_SERVICE_SSH_PASSWORD，使用 expect 自动输入 SSH 密码。"
    open_tunnel_with_expect
  else
    printf '%s\n' "如果提示密码，请输入服务器密码。"
    ssh \
      -f \
      -M \
      -S "$CONTROL_SOCKET" \
      -o StrictHostKeyChecking=no \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "${LOCAL_HOST}:${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
      -p "$SSH_PORT" \
      "${SSH_USER}@${SSH_HOST}" \
      -N
  fi

  TUNNEL_STARTED=1
  printf '%s\n' "SSH 隧道已建立。"
}

run_app() {
  cd "$REPO_ROOT"
  export FACE_SERVICE_BASE_URL="$BASE_URL"

  if command -v uv >/dev/null 2>&1; then
    exec uv run python main.py
  fi

  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    exec "$REPO_ROOT/.venv/bin/python" main.py
  fi

  printf '%s\n' "未找到 uv，也未找到 .venv/bin/python，请先安装依赖。" >&2
  exit 1
}

ensure_tunnel
run_app
