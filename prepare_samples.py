"""Prepare eight labeled CoatingVision examples for file replay and visual checks."""

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image


REVISION = "19d857658207491c5711625017c0db9a14233965"
BASE_URL = (
    f"https://raw.githubusercontent.com/vigsam-coder/CoatingVision/{REVISION}"
    "/dataset/Segmentation"
)
DEFAULT_DEST = Path(__file__).resolve().parent / "data" / "coatingvision-demo"
# Source Git blob IDs for image and annotation at the pinned revision.
SAMPLES = [
    ("train", "sample1", "30ec14d87f41f70fc82ddf714a3ea379819063fd", "3533742db68a509d66383fd214f65edc2c9456b3"),
    ("train", "sample2", "ad9af59eecfcffc921c5572c7381ed2995504bda", "4197c876f00f47f168d7886967d9d95f9b076b61"),
    ("train", "sample3", "4c70aff1a43b2a3faf0004656e42d29938bb57aa", "08881be2fe423eeb69c8991b959a384437d87d12"),
    ("train", "sample4", "830a1847d4b85565133f59b2e8158b5d3b535f5f", "a559576c9603f2b704ab83ab6b571ee9a0802d03"),
    ("val", "sample5", "01bfbad10d3a4f6c8b258c6153c6dbbb62518db8", "0a50e0d64b0664730c51a161e6b7fcfb6c49c513"),
    ("val", "sample6", "d0855c886ae6ba50a50e26dad133c83e030dd206", "ad15ca212bcdd70b3febe921aed33e12351a550b"),
    ("val", "sample7", "f5040a0a07042f8eea25af099840d1bae7b62f69", "8972f67fcb074d316d2ce38bb79333e475547eca"),
    ("val", "sample8", "57ff78942030fc20046ba22c094bec4701538c62", "1119ea10932a1aa5c5705986bdebad6421855e4b"),
]


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def blob_id(payload):
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def fetch_checked(destination, relative, expected):
    local = destination / "source" / relative
    if local.is_file():
        cached = local.read_bytes()
        if blob_id(cached) == expected:
            return cached
    with urlopen(f"{BASE_URL}/{relative}", timeout=25) as response:
        content = response.read()
    if blob_id(content) != expected:
        raise ValueError(f"Source integrity check failed for {relative}")
    atomic_write(local, content)
    return content


def write_png(path, image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    atomic_write(path, stream.getvalue())


def prepare_one(destination, sample):
    split, name, image_id, mask_id = sample
    raw = fetch_checked(destination, f"{split}/images/{name}.jpg", image_id)
    label = fetch_checked(destination, f"{split}/masks/{name}.png", mask_id)
    with Image.open(io.BytesIO(raw)) as photo:
        rgb = photo.convert("RGB")
    with Image.open(io.BytesIO(label)) as mask:
        if mask.mode != "RGBA" or mask.size != rgb.size:
            raise ValueError(f"{name}: expected a four-channel PNG aligned to its photo")
        # Paper's channel 0 (Pillow R). Alpha carries a fourth defect label.
        crack = np.asarray(mask.getchannel("R")) > 0
    pixels = np.array(rgb)
    pixels[crack] = np.rint(0.4 * pixels[crack] + 0.6 * np.array([255, 0, 0]))
    preview = Image.new("RGB", (rgb.width * 2, rgb.height))
    preview.paste(rgb, (0, 0))
    preview.paste(Image.fromarray(pixels), (rgb.width, 0))
    write_png(destination / "overlays" / f"{name}.png", preview)
    write_png(destination / "crack_masks" / f"{name}.png", Image.fromarray(crack.astype("uint8") * 255))
    atomic_write(destination / "frames" / f"{name}.jpg", raw)
    return {
        "sample_id": name,
        "source_split": split,
        "image": f"frames/{name}.jpg",
        "crack_mask": f"crack_masks/{name}.png",
        "overlay": f"overlays/{name}.png",
        "width": rgb.width,
        "height": rgb.height,
        "crack_pixels": int(crack.sum()),
        "source_revision": REVISION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda sample: prepare_one(args.dest, sample), SAMPLES))
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    manifest = args.dest / "manifest.csv"
    atomic_write(manifest, table.getvalue().encode("utf-8"))
    print(f"Prepared {len(rows)} aligned pairs. Manifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Sample preparation failed: {error}. Re-run to resume from verified files.")
