import hashlib
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Dict, Iterable, Sequence

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TrainingSnapshot:
    class_names: Sequence[str]
    sample_count: int
    identity_count: int
    dataset_signature: str
    class_samples: Dict[str, tuple[Path, ...]]


def build_training_snapshot(faces_root: Path) -> TrainingSnapshot:
    if not faces_root.exists():
        return TrainingSnapshot([], 0, 0, "", {})

    classes: Dict[str, list[Path]] = {}
    for identity_dir in sorted(faces_root.iterdir(), key=lambda p: p.name):
        if not identity_dir.is_dir():
            continue
        if identity_dir.name.startswith("."):
            continue

        samples = _collect_samples(identity_dir)
        if not samples:
            continue

        classes[identity_dir.name] = samples

    class_names = sorted(classes.keys())
    sample_count = sum(len(classes[name]) for name in class_names)
    identity_count = len(class_names)
    class_samples = {name: tuple(classes[name]) for name in class_names}
    dataset_signature = _compute_signature(class_samples)

    return TrainingSnapshot(
        class_names=class_names,
        sample_count=sample_count,
        identity_count=identity_count,
        dataset_signature=dataset_signature,
        class_samples=class_samples,
    )


def stage_imagefolder_dataset(snapshot: TrainingSnapshot, staging_root: Path) -> Path:
    staging_root.mkdir(parents=True, exist_ok=True)
    for class_name in snapshot.class_names:
        dest_dir = staging_root / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in snapshot.class_samples.get(class_name, ()):
            copy2(src, dest_dir / src.name)
    return staging_root


def _collect_samples(identity_dir: Path) -> list[Path]:
    samples: list[Path] = []
    for candidate in sorted(identity_dir.iterdir(), key=lambda p: p.name):
        if not candidate.is_file():
            continue
        if candidate.name.startswith("."):
            continue
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        samples.append(candidate)
    return samples


def _compute_signature(class_samples: Dict[str, tuple[Path, ...]]) -> str:
    if not class_samples:
        return ""

    parts: list[str] = []
    for name in sorted(class_samples):
        file_names = "|".join(p.name for p in class_samples[name])
        digest = hashlib.sha1(file_names.encode("utf-8")).hexdigest()
        parts.append(f"{name}:{len(class_samples[name])}:{digest}")
    return ";".join(parts)
