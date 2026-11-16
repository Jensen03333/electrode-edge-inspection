"""Save up to three labeled examples selected by disclosed, reproducible error rules."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark_model import cpu_session, infer, input_spec, read_manifest, read_mask


def load_scores(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["dataset_file"]: row for row in csv.DictReader(stream)}


def recall(row):
    tp, fn = int(row["tp_px"]), int(row["fn_px"])
    return tp / (tp + fn) if tp + fn else None


def draw_errors(image, truth, pred):
    gray = np.asarray(image.convert("L"))
    result = np.repeat(gray[:, :, None], 3, axis=2)
    for mask, color in ((truth & pred, (32, 220, 135)), (truth & ~pred, (255, 62, 58)),
                        (pred & ~truth, (98, 165, 255))):
        result[mask] = np.round(0.25 * result[mask] + 0.75 * np.array(color)).astype(np.uint8)
    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/full"))
    parser.add_argument("--evaluation", type=Path, default=Path("data/full/test.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--fp32", type=Path, default=Path("models/fp32.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("models/int8.onnx"))
    parser.add_argument("--out", type=Path, default=Path("reports/errors"))
    args = parser.parse_args()
    measurement = json.loads((args.benchmark / "comparison.json").read_text(encoding="utf-8"))
    threshold = measurement["logit_threshold"]
    index = {r["file"]: r for r in read_manifest(args.evaluation, args.data_root, masks=True)}
    fp, integer = (load_scores(args.benchmark / f"{tag}.csv") for tag in ("fp32", "int8"))
    if set(fp) != set(integer) or set(fp) != set(index):
        raise ValueError("Score tables and evaluation manifest must cover the same frames")
    positives = sorted(k for k in index if int(fp[k]["positive_frame"]))
    negatives = sorted(k for k in index if not int(fp[k]["positive_frame"]))
    cases = []
    if positives:
        cases.append(("lowest-int8-recall", min(positives, key=lambda k: recall(integer[k]))))
        cases.append(("largest-recall-loss", max(positives, key=lambda k: recall(fp[k])-recall(integer[k]))))
    if negatives:
        cases.append(("largest-negative-alarm", max(negatives, key=lambda k: int(integer[k]["fp_px"]))))
    sessions = [cpu_session(path, 1) for path in (args.fp32, args.int8)]
    args.out.mkdir(parents=True, exist_ok=True)
    for rule, key in cases:
        path = args.data_root / key
        truth = read_mask(args.data_root / index[key]["mask"])
        with Image.open(path) as photo:
            raw = photo.convert("RGB")
        panels = [raw] + [draw_errors(raw, truth, infer(s, input_spec(s), path, threshold)[0]) for s in sessions]
        width, height = raw.size
        mosaic = Image.new("RGB", (width * 3, height + 53), (26, 30, 37))
        pen = ImageDraw.Draw(mosaic)
        for i, (label, panel) in enumerate(zip(("Source", "FP32", "INT8"), panels)):
            mosaic.paste(panel, (i * width, 23))
            pen.text((i * width + 6, 4), label, fill="white")
        pen.text((6, height + 30), f"{rule} | {key} | green: hit    red: missed    blue: false alarm", fill="white")
        mosaic.save(args.out / f"{rule}.png")
        print(rule, key)


if __name__ == "__main__":
    main()
