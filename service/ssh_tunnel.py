from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_local_target(hostname: str | None) -> bool:
    return hostname in {"127.0.0.1", "localhost"}


@dataclass
class TunnelBootstrapResult:
    manager: Optional["FaceServiceSshTunnelManager"]
    message: str


class FaceServiceSshTunnelManager:
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_port: int,
        local_host: str,
        local_port: int,
        remote_host: str,
        remote_port: int,
        key_path: str | None = None,
        strict_host_key_checking: str = "no",
    ) -> None:
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.local_host = local_host
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.key_path = key_path
        self.strict_host_key_checking = strict_host_key_checking
        self.process: subprocess.Popen[str] | None = None
        self.started_by_app = False

    @classmethod
    def from_environment(cls) -> Optional["FaceServiceSshTunnelManager"]:
        base_url = os.getenv("FACE_SERVICE_BASE_URL", "http://127.0.0.1:18000").rstrip("/")
        parsed = urlparse(base_url)
        if not _is_local_target(parsed.hostname):
            return None

        ssh_host = os.getenv("FACE_SERVICE_SSH_HOST", "").strip()
        ssh_user = os.getenv("FACE_SERVICE_SSH_USER", "").strip()
        if not ssh_host or not ssh_user:
            return None

        if not _env_flag("FACE_SERVICE_SSH_AUTOSTART", True):
            return None

        local_host = os.getenv("FACE_SERVICE_SSH_LOCAL_HOST", parsed.hostname or "127.0.0.1").strip() or "127.0.0.1"
        local_port = int(os.getenv("FACE_SERVICE_SSH_LOCAL_PORT", str(parsed.port or 18000)))
        remote_host = os.getenv("FACE_SERVICE_SSH_REMOTE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        remote_port = int(os.getenv("FACE_SERVICE_SSH_REMOTE_PORT", "8000"))
        ssh_port = int(os.getenv("FACE_SERVICE_SSH_PORT", "22"))
        key_path = os.getenv("FACE_SERVICE_SSH_KEY_PATH", "").strip() or None
        strict_host_key_checking = os.getenv("FACE_SERVICE_SSH_STRICT_HOST_KEY_CHECKING", "no").strip() or "no"

        return cls(
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            local_host=local_host,
            local_port=local_port,
            remote_host=remote_host,
            remote_port=remote_port,
            key_path=key_path,
            strict_host_key_checking=strict_host_key_checking,
        )

    def is_local_port_open(self) -> bool:
        try:
            with socket.create_connection((self.local_host, self.local_port), timeout=0.4):
                return True
        except OSError:
            return False

    def build_command(self) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"StrictHostKeyChecking={self.strict_host_key_checking}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.ssh_port),
            "-L",
            f"{self.local_host}:{self.local_port}:{self.remote_host}:{self.remote_port}",
        ]
        if self.key_path:
            command.extend(["-i", self.key_path])
        command.extend([f"{self.ssh_user}@{self.ssh_host}", "-N"])
        return command

    def start(self, wait_seconds: float = 4.0) -> str:
        if self.is_local_port_open():
            return (
                f"检测到本地 {self.local_host}:{self.local_port} 已监听，"
                "直接复用现有 SSH 隧道"
            )

        try:
            process = subprocess.Popen(
                self.build_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            return f"SSH 隧道启动失败：{exc}"

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.is_local_port_open():
                self.process = process
                self.started_by_app = True
                return (
                    f"已自动启动 SSH 隧道："
                    f"{self.local_host}:{self.local_port} -> {self.remote_host}:{self.remote_port}"
                )

            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                stderr = stderr.strip()
                if stderr:
                    return f"SSH 隧道启动失败：{stderr}"
                return "SSH 隧道启动失败：ssh 进程提前退出"

            time.sleep(0.1)

        process.terminate()
        try:
            _, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=1)

        stderr = stderr.strip()
        if stderr:
            return f"SSH 隧道启动失败：{stderr}"
        return "SSH 隧道启动失败：等待本地转发端口就绪超时"

    def stop(self) -> None:
        if not self.started_by_app or self.process is None:
            return
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


def bootstrap_face_service_tunnel() -> TunnelBootstrapResult:
    manager = FaceServiceSshTunnelManager.from_environment()
    if manager is None:
        return TunnelBootstrapResult(manager=None, message="")

    message = manager.start()
    if message.startswith("SSH 隧道启动失败"):
        return TunnelBootstrapResult(manager=None, message=message)
    return TunnelBootstrapResult(manager=manager, message=message)
