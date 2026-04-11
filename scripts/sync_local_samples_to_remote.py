from __future__ import annotations

import argparse
from pathlib import Path

from service.remote_sync import (
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST_PATH,
    reconcile_remote_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile local cropped face samples in dataset/full with the remote embedding service."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local dataset root, default: dataset/full",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Remote face service base URL, default: http://127.0.0.1:18000",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path used to skip unchanged identities.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Only sync one identity directory, for example --name 张三",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="HTTP timeout in seconds, default: 8",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild and re-upload all matching identities.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be changed, without sending requests.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one remote operation fails.",
    )
    parser.add_argument(
        "--keep-remote-missing",
        action="store_true",
        help="Do not delete remote identities that are missing locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = reconcile_remote_samples(
        dataset_root=args.dataset_root,
        base_url=args.base_url,
        manifest_path=args.manifest_path,
        target_name=args.name,
        timeout=args.timeout,
        force=args.force,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        keep_remote_missing=args.keep_remote_missing,
        progress=print,
    )
    print(
        f"[summary] rebuilt={summary.rebuilt_identities}/{summary.planned_rebuilds} "
        f"uploaded_files={summary.uploaded_files} skipped={summary.skipped_identities} "
        f"deleted={summary.deleted_identities} failed={summary.failed} dry_run={summary.dry_run}"
    )
    if summary.error:
        print(f"[error] {summary.error}")
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
