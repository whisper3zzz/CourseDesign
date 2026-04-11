from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced CelebA subset for local MindSpore training."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/celeba/raw"),
        help="CelebA raw root containing img_align_celeba/ and annotation files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/celeba/subsets/course_v1"),
        help="Output directory for the prepared subset.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional override for the aligned image directory.",
    )
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=None,
        help="Optional override for identity_CelebA.txt.",
    )
    parser.add_argument(
        "--partition-file",
        type=Path,
        default=None,
        help="Optional override for list_eval_partition.txt.",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=50,
        help="Maximum number of identities to keep.",
    )
    parser.add_argument(
        "--train-per-identity",
        type=int,
        default=20,
        help="Training images per identity.",
    )
    parser.add_argument(
        "--val-per-identity",
        type=int,
        default=5,
        help="Validation images per identity.",
    )
    parser.add_argument(
        "--test-per-identity",
        type=int,
        default=5,
        help="Test images per identity.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("balanced", "official"),
        default="balanced",
        help="Use a custom balanced split or the official CelebA partition file.",
    )
    parser.add_argument(
        "--selection",
        choices=("top", "random"),
        default="top",
        help="Pick the most populated identities or sample identities randomly.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to place images into the subset directories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for identity and image selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def resolve_images_dir(raw_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = (
        raw_root / "img_align_celeba",
        raw_root / "img_align_celeba" / "img_align_celeba",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find aligned image directory. Expected raw/img_align_celeba/."
    )


def resolve_annotation_file(raw_root: Path, override: Path | None, name: str) -> Path:
    path = override if override is not None else raw_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    return path


def read_identity_groups(identity_file: Path) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    with identity_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            try:
                identity = int(parts[1])
            except ValueError:
                continue
            grouped[identity].append(image_name)
    return grouped


def read_partition_map(partition_file: Path) -> dict[str, int]:
    partitions: dict[str, int] = {}
    with partition_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                partitions[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return partitions


def choose_identities(
    groups: dict[int, list[str]],
    required_count: int,
    max_identities: int,
    selection: str,
    seed: int,
) -> list[tuple[int, list[str]]]:
    eligible = [
        (identity, sorted(images))
        for identity, images in groups.items()
        if len(images) >= required_count
    ]
    if not eligible:
        raise RuntimeError(
            f"No identity has at least {required_count} usable images."
        )
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    if selection == "random":
        rng = random.Random(seed)
        rng.shuffle(eligible)
    return eligible[:max_identities]


def split_images_balanced(
    image_names: list[str],
    train_per_identity: int,
    val_per_identity: int,
    test_per_identity: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    names = list(image_names)
    rng.shuffle(names)
    train_end = train_per_identity
    val_end = train_end + val_per_identity
    test_end = val_end + test_per_identity
    return {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:test_end],
    }


def split_images_official(
    image_names: list[str],
    partitions: dict[str, int],
    train_per_identity: int,
    val_per_identity: int,
    test_per_identity: int,
    rng: random.Random,
) -> dict[str, list[str]] | None:
    pools = {"train": [], "val": [], "test": []}
    for image_name in image_names:
        part = partitions.get(image_name)
        if part == 0:
            pools["train"].append(image_name)
        elif part == 1:
            pools["val"].append(image_name)
        elif part == 2:
            pools["test"].append(image_name)
    for split_name, required in (
        ("train", train_per_identity),
        ("val", val_per_identity),
        ("test", test_per_identity),
    ):
        if len(pools[split_name]) < required:
            return None
        rng.shuffle(pools[split_name])
        pools[split_name] = pools[split_name][:required]
    return pools


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        has_content = any(output_root.iterdir())
        if has_content and not overwrite:
            raise FileExistsError(
                f"{output_root} already exists and is not empty. Use --overwrite to replace it."
            )
        if has_content and overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    required_count = (
        args.train_per_identity + args.val_per_identity + args.test_per_identity
    )

    images_dir = resolve_images_dir(args.raw_root, args.images_dir)
    identity_file = resolve_annotation_file(
        args.raw_root, args.identity_file, "identity_CelebA.txt"
    )
    partition_file = None
    partitions: dict[str, int] = {}
    if args.split_mode == "official":
        partition_file = resolve_annotation_file(
            args.raw_root, args.partition_file, "list_eval_partition.txt"
        )
        partitions = read_partition_map(partition_file)

    identity_groups = read_identity_groups(identity_file)
    chosen = choose_identities(
        groups=identity_groups,
        required_count=required_count,
        max_identities=args.max_identities,
        selection=args.selection,
        seed=args.seed,
    )

    ensure_output_root(args.output_root, overwrite=args.overwrite)
    (args.output_root / "meta").mkdir(parents=True, exist_ok=True)

    selected_records: list[dict[str, object]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    for identity, image_names in chosen:
        if args.split_mode == "official":
            split_map = split_images_official(
                image_names=image_names,
                partitions=partitions,
                train_per_identity=args.train_per_identity,
                val_per_identity=args.val_per_identity,
                test_per_identity=args.test_per_identity,
                rng=rng,
            )
            if split_map is None:
                continue
        else:
            split_map = split_images_balanced(
                image_names=image_names,
                train_per_identity=args.train_per_identity,
                val_per_identity=args.val_per_identity,
                test_per_identity=args.test_per_identity,
                rng=rng,
            )

        folder_name = f"id_{identity:05d}"
        record = {
            "celeb_id": identity,
            "folder_name": folder_name,
            "counts": {
                "train": len(split_map["train"]),
                "val": len(split_map["val"]),
                "test": len(split_map["test"]),
            },
        }

        for split_name, images in split_map.items():
            for image_name in images:
                src = images_dir / image_name
                if not src.is_file():
                    raise FileNotFoundError(f"Missing source image: {src}")
                dst = args.output_root / split_name / folder_name / image_name
                place_image(src, dst, args.copy_mode)
                split_counts[split_name] += 1

        selected_records.append(record)

    if not selected_records:
        raise RuntimeError(
            "No identities satisfied the requested split constraints. "
            "Try reducing per-identity counts or using balanced split mode."
        )

    identity_map_path = args.output_root / "meta" / "identity_map.tsv"
    with identity_map_path.open("w", encoding="utf-8") as handle:
        handle.write("folder_name\tceleb_id\ttrain\tval\ttest\n")
        for record in selected_records:
            counts = record["counts"]
            handle.write(
                f"{record['folder_name']}\t{record['celeb_id']}\t"
                f"{counts['train']}\t{counts['val']}\t{counts['test']}\n"
            )

    summary = {
        "source": {
            "raw_root": str(args.raw_root),
            "images_dir": str(images_dir),
            "identity_file": str(identity_file),
            "partition_file": str(partition_file) if partition_file else None,
        },
        "config": {
            "split_mode": args.split_mode,
            "selection": args.selection,
            "copy_mode": args.copy_mode,
            "max_identities": args.max_identities,
            "train_per_identity": args.train_per_identity,
            "val_per_identity": args.val_per_identity,
            "test_per_identity": args.test_per_identity,
            "seed": args.seed,
        },
        "result": {
            "selected_identities": len(selected_records),
            "split_counts": split_counts,
        },
    }
    summary_path = args.output_root / "meta" / "subset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("CelebA subset prepared successfully.")
    print(f"output_root: {args.output_root}")
    print(f"selected_identities: {len(selected_records)}")
    print(
        "split_counts: "
        f"train={split_counts['train']} "
        f"val={split_counts['val']} "
        f"test={split_counts['test']}"
    )
    print(f"identity_map: {identity_map_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
