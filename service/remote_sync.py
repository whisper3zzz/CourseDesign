from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_DATASET_ROOT = Path("dataset/full")
DEFAULT_MANIFEST_PATH = Path("temp/remote_sync_manifest.json")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class FileSignature:
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class IdentitySnapshot:
    name: str
    sample_paths: list[Path]
    signature: str


@dataclass(frozen=True)
class RemoteSyncSummary:
    ok: bool
    base_url: str
    dry_run: bool
    planned_rebuilds: int
    rebuilt_identities: int
    uploaded_files: int
    skipped_identities: int
    deleted_identities: int
    failed: int
    remote_identity_count: int
    local_identity_count: int
    error: str = ""

    @property
    def changed(self) -> bool:
        return (self.rebuilt_identities + self.deleted_identities) > 0

    def summary_line(self) -> str:
        return (
            f"rebuild={self.rebuilt_identities}/{self.planned_rebuilds} "
            f"upload={self.uploaded_files} skip={self.skipped_identities} "
            f"delete={self.deleted_identities} fail={self.failed}"
        )


class RemoteSyncError(RuntimeError):
    pass


class SyncManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload = {"version": 2, "identities": {}}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        identities = payload.get("identities")
        if not isinstance(identities, dict):
            identities = {}

        self.payload = {
            "version": 2,
            "identities": identities,
        }

    def get_identity_signature(self, name: str, base_url: str) -> str | None:
        entry = self.payload["identities"].get(name)
        if not isinstance(entry, dict):
            return None
        if entry.get("base_url") != base_url:
            return None
        signature = entry.get("signature")
        return signature if isinstance(signature, str) else None

    def mark_identity_synced(self, name: str, base_url: str, signature: str, file_count: int) -> None:
        self.payload["identities"][name] = {
            "base_url": base_url,
            "signature": signature,
            "file_count": file_count,
        }

    def remove_identity(self, name: str, base_url: str | None = None) -> None:
        entry = self.payload["identities"].get(name)
        if not isinstance(entry, dict):
            return
        if base_url is not None and entry.get("base_url") != base_url:
            return
        self.payload["identities"].pop(name, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> FileSignature:
    stat = path.stat()
    return FileSignature(
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(path),
    )


def build_identity_snapshot(dataset_root: Path, identity_dir: Path) -> IdentitySnapshot | None:
    signature_items: list[str] = []
    sample_paths: list[Path] = []

    for sample_path in sorted(identity_dir.iterdir()):
        if not sample_path.is_file() or sample_path.name.startswith("."):
            continue
        if sample_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        sample_sig = file_signature(sample_path)
        relative_path = sample_path.relative_to(dataset_root).as_posix()
        signature_items.append(
            f"{relative_path}:{sample_sig.size}:{sample_sig.mtime_ns}:{sample_sig.sha256}"
        )
        sample_paths.append(sample_path)

    if not sample_paths:
        return None

    signature = hashlib.sha256("\n".join(signature_items).encode("utf-8")).hexdigest()
    return IdentitySnapshot(
        name=identity_dir.name,
        sample_paths=sample_paths,
        signature=signature,
    )


def scan_local_identities(dataset_root: Path, target_name: str) -> dict[str, IdentitySnapshot]:
    identities: dict[str, IdentitySnapshot] = {}
    if not dataset_root.exists():
        return identities

    for identity_dir in sorted(dataset_root.iterdir()):
        if not identity_dir.is_dir() or identity_dir.name.startswith("."):
            continue
        if target_name and identity_dir.name != target_name:
            continue

        snapshot = build_identity_snapshot(dataset_root, identity_dir)
        if snapshot is not None:
            identities[snapshot.name] = snapshot

    return identities


def check_health(session: requests.Session, base_url: str, timeout: float) -> dict:
    response = session.get(f"{base_url}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_remote_identities(session: requests.Session, base_url: str, timeout: float) -> dict[str, int]:
    response = session.get(f"{base_url}/identities", timeout=timeout)
    if response.status_code == 404:
        raise RemoteSyncError("远端服务版本过旧，不支持 /identities，请先更新并重启服务端")
    response.raise_for_status()

    payload = response.json()
    identities = payload.get("identities", [])
    remote: dict[str, int] = {}
    for item in identities:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        sample_count = item.get("sample_count", 0)
        if isinstance(name, str):
            remote[name] = int(sample_count)
    return remote


def delete_remote_identity(
    session: requests.Session,
    base_url: str,
    name: str,
    timeout: float,
) -> dict:
    response = session.post(
        f"{base_url}/delete_identity",
        data={"name": name},
        timeout=timeout,
    )
    if response.status_code == 404:
        raise RemoteSyncError("远端服务版本过旧，不支持 /delete_identity，请先更新并重启服务端")
    response.raise_for_status()
    return response.json()


def upload_sample(
    session: requests.Session,
    base_url: str,
    name: str,
    sample_path: Path,
    timeout: float,
) -> dict:
    content_type = mimetypes.guess_type(sample_path.name)[0] or "application/octet-stream"
    with sample_path.open("rb") as photo_file:
        response = session.post(
            f"{base_url}/register",
            data={"name": name},
            files={"photo": (sample_path.name, photo_file, content_type)},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def reconcile_remote_samples(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    base_url: str = DEFAULT_BASE_URL,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    target_name: str = "",
    timeout: float = 8.0,
    force: bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
    keep_remote_missing: bool = False,
    progress: ProgressCallback | None = None,
) -> RemoteSyncSummary:
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    base_url = base_url.rstrip("/")
    target_name = target_name.strip()

    if not dataset_root.exists():
        return RemoteSyncSummary(
            ok=False,
            base_url=base_url,
            dry_run=dry_run,
            planned_rebuilds=0,
            rebuilt_identities=0,
            uploaded_files=0,
            skipped_identities=0,
            deleted_identities=0,
            failed=1,
            remote_identity_count=0,
            local_identity_count=0,
            error=f"未找到本地样本目录：{dataset_root}，已跳过远端同步",
        )

    local_identities = scan_local_identities(dataset_root, target_name)
    manifest = SyncManifest(manifest_path)
    session = requests.Session()

    try:
        health_payload = check_health(session, base_url, timeout)
        _emit(
            progress,
            f"远端服务在线：status={health_payload.get('status')} "
            f"identities={health_payload.get('identity_count')} "
            f"embeddings={health_payload.get('embedding_count')}",
        )
        remote_identities = fetch_remote_identities(session, base_url, timeout)
    except (requests.RequestException, RemoteSyncError) as exc:
        return RemoteSyncSummary(
            ok=False,
            base_url=base_url,
            dry_run=dry_run,
            planned_rebuilds=0,
            rebuilt_identities=0,
            uploaded_files=0,
            skipped_identities=0,
            deleted_identities=0,
            failed=1,
            remote_identity_count=0,
            local_identity_count=len(local_identities),
            error=str(exc),
        )

    local_names = set(local_identities)
    remote_names = set(remote_identities)

    planned_rebuilds = 0
    rebuilt_identities = 0
    uploaded_files = 0
    skipped_identities = 0
    deleted_identities = 0
    failed = 0

    for name in sorted(local_names):
        snapshot = local_identities[name]
        previous_signature = manifest.get_identity_signature(name, base_url)
        remote_present = name in remote_names
        unchanged = previous_signature == snapshot.signature and remote_present

        if unchanged and not force:
            _emit(progress, f"远端已是最新：{name} ({len(snapshot.sample_paths)} 张)")
            skipped_identities += 1
            continue

        planned_rebuilds += 1
        if dry_run:
            action = "重建" if remote_present else "创建"
            _emit(progress, f"启动预览：将{action}远端身份 {name} ({len(snapshot.sample_paths)} 张)")
            continue

        try:
            if remote_present:
                payload = delete_remote_identity(session, base_url, name, timeout)
                _emit(progress, f"已删除远端旧身份：{name} removed={payload.get('removed')}")

            for sample_path in snapshot.sample_paths:
                payload = upload_sample(session, base_url, name, sample_path, timeout)
                uploaded_files += 1
                _emit(
                    progress,
                    f"已同步 {sample_path.relative_to(dataset_root).as_posix()} "
                    f"(remote samples={payload.get('sample_count')})",
                )

            manifest.mark_identity_synced(name, base_url, snapshot.signature, len(snapshot.sample_paths))
            rebuilt_identities += 1
        except (requests.RequestException, RemoteSyncError) as exc:
            _emit(progress, f"同步失败 {name}: {exc}")
            failed += 1
            if fail_fast:
                break

    deletable_remote_names: set[str] = set()
    if not keep_remote_missing:
        if target_name:
            if target_name in remote_names and target_name not in local_names:
                deletable_remote_names.add(target_name)
        else:
            deletable_remote_names = remote_names - local_names

    for name in sorted(deletable_remote_names):
        if dry_run:
            _emit(progress, f"启动预览：将删除远端多余身份 {name}")
            continue

        try:
            payload = delete_remote_identity(session, base_url, name, timeout)
            manifest.remove_identity(name, base_url)
            deleted_identities += 1
            _emit(progress, f"已删除远端多余身份：{name} removed={payload.get('removed')}")
        except (requests.RequestException, RemoteSyncError) as exc:
            _emit(progress, f"删除远端身份失败 {name}: {exc}")
            failed += 1
            if fail_fast:
                break

    if not dry_run:
        manifest.save()

    return RemoteSyncSummary(
        ok=(failed == 0),
        base_url=base_url,
        dry_run=dry_run,
        planned_rebuilds=planned_rebuilds,
        rebuilt_identities=rebuilt_identities,
        uploaded_files=uploaded_files,
        skipped_identities=skipped_identities,
        deleted_identities=deleted_identities,
        failed=failed,
        remote_identity_count=len(remote_identities),
        local_identity_count=len(local_identities),
        error="" if failed == 0 else f"同步过程中有 {failed} 个失败项，请查看上方日志",
    )
