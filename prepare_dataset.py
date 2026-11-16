"""Prepare four source-grouped surface-crack splits from the official CoatingVision ZIP."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import random
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


ARCHIVE_MD5 = "7cbbf14725f9762a64a81a12b858a2b4"
DOI = "10.6084/m9.figshare.29260121.v1"
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.15, "calibration": 0.10, "test": 0.15}


@dataclass
class Record:
    name: str
    group: str
    original: str
    photo_path: str
    mask_path: str
    positive: bool
    image_hash: str
    mask_hash: str


def source_group(original):
    # Both "center" and "center-1" sequences of R1-1100um stay together.
    match = re.fullmatch(r"(R\d+-\d+um)-.+_frame_\d+_patch_\d+\.png", original)
    if not match:
        raise ValueError(f"Cannot identify source group for {original!r}")
    return match.group(1)


def split_groups(records, seed):
    """Keep intact acquisition groups, balancing image and crack-negative counts."""
    sizes = Counter(r.group for r in records)
    negatives = Counter(r.group for r in records if not r.positive)
    names = sorted(sizes)
    if len(names) < 4:
        raise ValueError("At least four independent acquisition groups are required.")
    candidates = list(itertools.permutations(names, 3))
    random.Random(seed).shuffle(candidates)
    total = len(records)
    total_negatives = sum(negatives.values())

    def score(held_out):
        by_split = dict(zip(("val", "calibration", "test"), [(g,) for g in held_out]))
        by_split["train"] = [name for name in names if name not in held_out]
        error = 0.0
        for split, groups in by_split.items():
            fraction = SPLIT_FRACTIONS[split]
            count = sum(sizes[g] for g in groups)
            error += abs(count / total - fraction)
            if total_negatives:
                negative_count = sum(negatives[g] for g in groups)
                error += 0.5 * abs(negative_count / total_negatives - fraction)
                if negative_count == 0 or negative_count == count:
                    error += 1
        return error

    selected = min(candidates, key=score)
    result = {group: "train" for group in names}
    result.update(zip(selected, ("val", "calibration", "test")))
    return result


def crack_mask(archive, mask_path):
    with Image.open(io.BytesIO(archive.read(mask_path))) as label:
        if label.mode != "RGBA":
            raise ValueError(f"{mask_path}: expected four native PNG annotation channels")
        crack = np.asarray(label.getchannel("R")) > 0
    return crack


def inspect_records(archive):
    members = archive.namelist()
    photos = {}
    masks = {}
    for name in members:
        path = PurePosixPath(name)
        parent = path.parent.as_posix().lower()
        if parent.endswith("segmentation/images") and path.suffix.lower() in {".jpg", ".jpeg"}:
            photos[path.stem] = name
        if parent.endswith("segmentation/masks") and path.suffix.lower() == ".png":
            masks[path.stem] = name
    metadata_path = next(name for name in members if name.lower().endswith("classification/labels.csv"))
    with archive.open(metadata_path) as data:
        metadata = list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig")))
    if not photos or photos.keys() != masks.keys():
        raise ValueError(f"Unmatched segmentation sources: {len(photos)} photos vs {len(masks)} masks.")
    records = []
    metadata_names = set()
    classification_disagreements = 0
    for line in metadata:
        name = PurePosixPath(line["file_name"]).stem
        if not re.fullmatch(r"image_\d+", name) or name in metadata_names:
            raise ValueError(f"Unexpected or duplicate image ID {name!r}")
        metadata_names.add(name)
        if name not in photos:
            continue
        with Image.open(io.BytesIO(archive.read(photos[name]))) as image:
            rgb = image.convert("RGB")
            dimensions = f"{rgb.width}x{rgb.height}:".encode("ascii")
            image_hash = hashlib.sha256(dimensions + rgb.tobytes()).hexdigest()
        crack = crack_mask(archive, masks[name])
        if crack.shape != (rgb.height, rgb.width):
            raise ValueError(f"{name}: image and mask dimensions differ")
        original = line["original_file_name"]
        positive = bool(crack.any())
        if line.get("Surface_Crack") in {"0", "1"} and positive != (line["Surface_Crack"] == "1"):
            classification_disagreements += 1
        records.append(Record(name, source_group(original), original, photos[name], masks[name],
                              positive, image_hash, hashlib.sha256(crack.tobytes()).hexdigest()))
    if len(records) != len(photos):
        raise ValueError("Some segmentation pairs have no acquisition metadata.")
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record.image_hash].append(record)
    unique = []
    conflicts = []
    for group in by_hash.values():
        if len({record.mask_hash for record in group}) > 1:
            conflicts.extend(record.name for record in group)
        else:
            unique.append(min(group, key=lambda record: record.name))
    originals = [r.original for r in unique]
    if len(set(originals)) != len(originals):
        raise ValueError("Distinct images share a source patch ID; inspect the archive before proceeding.")
    return unique, {
        "archive_pairs": len(records),
        "mask_presence_differs_from_classification_csv": classification_disagreements,
        "exact_pixel_duplicates_removed": len(records) - len(unique) - len(conflicts),
        "conflicting_duplicate_records_excluded": conflicts,
    }


def write_dataset(archive, output, seed):
    records, stats = inspect_records(archive)
    assigned = split_groups(records, seed)
    split_summary = {}
    for split in SPLIT_FRACTIONS:
        image_dir = output / split / "images"
        mask_dir = output / split / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        rows = sorted((r for r in records if assigned[r.group] == split), key=lambda r: r.name)
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["file", "mask", "group", "original", "crack_pixels"])
            writer.writeheader()
            for record in rows:
                image_path = image_dir / (record.name + ".jpg")
                mask_path = mask_dir / (record.name + ".png")
                image_path.write_bytes(archive.read(record.photo_path))
                crack = crack_mask(archive, record.mask_path)
                Image.fromarray(crack.astype("uint8") * 255).save(mask_path)
                writer.writerow({
                    "file": image_path.relative_to(output).as_posix(),
                    "mask": mask_path.relative_to(output).as_posix(),
                    "group": record.group,
                    "original": record.original,
                    "crack_pixels": int(crack.sum()),
                })
        split_summary[split] = {
            "images": len(rows), "crack_positive": sum(r.positive for r in rows),
            "crack_negative": sum(not r.positive for r in rows), "groups": sorted({r.group for r in rows}),
        }
    summary = dict(source_doi=DOI, archive_md5=ARCHIVE_MD5, seed=seed, **stats,
                   group_rule="prefix R<id>-<gap>um from original_file_name (not a factory roll ID)",
                   splits=split_summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def verify_manifests(root, summary):
    """Check that camera frames and labels exist and each acquisition has one role."""
    group_owner = {}
    patch_ids = set()
    for split, expected in summary["splits"].items():
        with (root / f"{split}.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != expected["images"]:
            raise ValueError(f"{split}: row count disagrees with summary")
        for row in rows:
            group = source_group(row["original"])
            if row["group"] != group or group_owner.setdefault(group, split) != split:
                raise ValueError(f"Invalid acquisition group in {split}: {row['original']}")
            if row["original"] in patch_ids:
                raise ValueError(f"Repeated original patch: {row['original']}")
            patch_ids.add(row["original"])
            if not (root / row["file"]).is_file() or not (root / row["mask"]).is_file():
                raise FileNotFoundError(f"{split}: missing image or annotation for {row['original']}")
        if sum(int(row["crack_pixels"]) > 0 for row in rows) != expected["crack_positive"]:
            raise ValueError(f"{split}: positive count disagrees with summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "data" / "full")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    checksum = hashlib.md5()
    with args.archive.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != ARCHIVE_MD5:
        parser.error("The archive differs from the official Figshare v1 checksum.")
    destination = args.dest.resolve()
    if destination.exists():
        manifest = destination / "summary.json"
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("seed") == args.seed and existing.get("archive_md5") == ARCHIVE_MD5:
                verify_manifests(destination, existing)
                print(json.dumps(existing, indent=2))
                return
        parser.error(f"Output already exists at {destination}. Choose a different --dest.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        staging = Path(work) / "prepared"
        staging.mkdir()
        with zipfile.ZipFile(args.archive) as archive:
            summary = write_dataset(archive, staging, args.seed)
        verify_manifests(staging, summary)
        os.replace(staging, destination)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
