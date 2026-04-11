import argparse
import csv
import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests


@dataclass
class PubFigEntry:
    person: str
    image_num: str
    url: str
    rect: Tuple[int, int, int, int]
    md5sum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced PubFig development subset.")
    parser.add_argument(
        "--manifest",
        default="data/pubfig/raw/dev_urls.txt",
        help="Path to PubFig dev_urls.txt manifest.",
    )
    parser.add_argument(
        "--people-file",
        default="data/pubfig/raw/dev_people.txt",
        help="Path to PubFig dev_people.txt list.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pubfig/dev_subset",
        help="Output directory for downloaded subset.",
    )
    parser.add_argument(
        "--num-people",
        type=int,
        default=30,
        help="Number of identities to download from the development set.",
    )
    parser.add_argument(
        "--images-per-person",
        type=int,
        default=20,
        help="Target number of successful cropped face images per identity.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Output face image size.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="HTTP timeout for each download request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling candidates per identity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if cropped output already exists.",
    )
    return parser.parse_args()


def read_people(people_file: Path) -> List[str]:
    people = []
    with people_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            people.append(stripped)
    return people


def parse_rect(value: str) -> Tuple[int, int, int, int]:
    parts = [int(float(item)) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"invalid rect: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def read_manifest(manifest_path: Path) -> Dict[str, List[PubFigEntry]]:
    grouped: Dict[str, List[PubFigEntry]] = {}
    with manifest_path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 5:
                continue
            entry = PubFigEntry(
                person=row[0].strip(),
                image_num=row[1].strip(),
                url=row[2].strip(),
                rect=parse_rect(row[3].strip()),
                md5sum=row[4].strip(),
            )
            grouped.setdefault(entry.person, []).append(entry)
    return grouped


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def decode_image(content: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def crop_face(image: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    face = image[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return face


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_report(report_path: Path, row: Sequence[str]) -> None:
    ensure_parent(report_path)
    with report_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def should_skip(output_file: Path, overwrite: bool) -> bool:
    return output_file.exists() and not overwrite


def md5_hex(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def stable_seed(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def select_people(
    people: List[str],
    grouped_entries: Dict[str, List[PubFigEntry]],
    limit: int,
    seed: int,
) -> List[str]:
    selected = [person for person in people if person in grouped_entries]
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def download_subset(
    selected_people: Iterable[str],
    grouped_entries: Dict[str, List[PubFigEntry]],
    output_dir: Path,
    image_size: int,
    images_per_person: int,
    timeout: float,
    seed: int,
    overwrite: bool,
) -> None:
    crops_dir = output_dir / "faces_112"
    report_path = output_dir / "download_report.tsv"
    summary_path = output_dir / "summary.tsv"
    metadata_path = output_dir / "subset_manifest.tsv"
    session = requests.Session()
    session.headers.update({"User-Agent": "CourseDesignPubFigDownloader/1.0"})
    session.trust_env = False

    if overwrite:
        for target in (report_path, summary_path, metadata_path):
            if target.exists():
                target.unlink()

    if not report_path.exists():
        append_report(
            report_path,
            ["person", "image_num", "status", "url", "output_file", "note"],
        )
    if not summary_path.exists():
        append_report(summary_path, ["person", "requested", "downloaded", "attempted"])
    if not metadata_path.exists():
        append_report(
            metadata_path,
            ["person", "image_num", "url", "rect", "source_md5", "downloaded_md5", "output_file"],
        )

    total_downloaded = 0
    total_requested = 0

    for person in selected_people:
        target_dir = crops_dir / sanitize_name(person)
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates = list(grouped_entries[person])
        random.Random(seed + stable_seed(person)).shuffle(candidates)
        downloaded = 0
        attempted = 0
        total_requested += images_per_person

        for entry in candidates:
            if downloaded >= images_per_person:
                break

            output_file = target_dir / f"{sanitize_name(person)}_{entry.image_num}.jpg"
            if should_skip(output_file, overwrite):
                downloaded += 1
                attempted += 1
                append_report(
                    report_path,
                    [entry.person, entry.image_num, "exists", entry.url, str(output_file), ""],
                )
                continue

            attempted += 1
            note = ""
            try:
                response = session.get(entry.url, timeout=timeout)
                if response.status_code != 200:
                    append_report(
                        report_path,
                        [entry.person, entry.image_num, f"http_{response.status_code}", entry.url, "", ""],
                    )
                    continue
                content = response.content
                image = decode_image(content)
                if image is None:
                    append_report(
                        report_path,
                        [entry.person, entry.image_num, "decode_failed", entry.url, "", ""],
                    )
                    continue
                face = crop_face(image, entry.rect)
                if face is None:
                    append_report(
                        report_path,
                        [entry.person, entry.image_num, "crop_failed", entry.url, "", ""],
                    )
                    continue
                resized = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)
                ensure_parent(output_file)
                if not cv2.imwrite(str(output_file), resized):
                    append_report(
                        report_path,
                        [entry.person, entry.image_num, "write_failed", entry.url, str(output_file), ""],
                    )
                    continue

                downloaded_md5 = md5_hex(content)
                if entry.md5sum and downloaded_md5.lower() != entry.md5sum.lower():
                    note = "md5_mismatch"

                append_report(
                    report_path,
                    [entry.person, entry.image_num, "downloaded", entry.url, str(output_file), note],
                )
                append_report(
                    metadata_path,
                    [
                        entry.person,
                        entry.image_num,
                        entry.url,
                        ",".join(str(v) for v in entry.rect),
                        entry.md5sum,
                        downloaded_md5,
                        str(output_file),
                    ],
                )
                downloaded += 1
                total_downloaded += 1
                time.sleep(0.1)
            except requests.RequestException as exc:
                append_report(
                    report_path,
                    [entry.person, entry.image_num, "request_failed", entry.url, "", str(exc)],
                )

        append_report(summary_path, [person, str(images_per_person), str(downloaded), str(attempted)])
        print(f"{person}: downloaded {downloaded}/{images_per_person} after {attempted} attempts")

    print(f"subset complete: downloaded {total_downloaded}/{total_requested} cropped faces")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    people_path = Path(args.people_file)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not people_path.exists():
        raise FileNotFoundError(f"people file not found: {people_path}")

    people = read_people(people_path)
    grouped_entries = read_manifest(manifest_path)
    selected_people = select_people(people, grouped_entries, args.num_people, args.seed)
    if not selected_people:
        raise RuntimeError("no identities selected from manifest")

    print(
        f"selected {len(selected_people)} identities, target {args.images_per_person} images per identity"
    )
    download_subset(
        selected_people=selected_people,
        grouped_entries=grouped_entries,
        output_dir=output_dir,
        image_size=args.image_size,
        images_per_person=args.images_per_person,
        timeout=args.timeout,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
