"""Run a deployed ONNX crack model on images without labels or a training environment."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_model import cpu_session, infer, input_spec


def inspect(source, model, out, threshold, threads):
    source, out = source.resolve(), out.resolve()
    if source.is_dir() and out.is_relative_to(source):
        raise ValueError("Choose an output directory outside the input image tree")
    root = source if source.is_dir() else source.parent
    candidates = source.rglob("*") if source.is_dir() else [source]
    frames = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames or not np.isfinite(threshold):
        raise ValueError("Expected input images and a finite model threshold")
    session = cpu_session(model, threads)
    spec = input_spec(session)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inspection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "width", "height", "crack_pixels", "service_ms", "mask", "overlay"))
        for frame in frames:
            pred, duration = infer(session, spec, frame, threshold)
            name = frame.relative_to(root)
            mask_path = out / "masks" / name.with_name(name.name + ".png")
            overlay_path = out / "overlays" / name.with_name(name.name + ".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8) * 255).save(mask_path)
            with Image.open(frame) as original:
                overlay = np.array(original.convert("RGB"))
            overlay[pred] = (0.30 * overlay[pred] + 0.70 * np.array([255, 64, 64])).astype(np.uint8)
            Image.fromarray(overlay).save(overlay_path)
            writer.writerow((name.as_posix(), pred.shape[1], pred.shape[0], int(pred.sum()),
                             round(duration, 4), mask_path.relative_to(out).as_posix(),
                             overlay_path.relative_to(out).as_posix()))
    print(f"{len(frames)} frame(s) inspected. Results: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    inspect(args.input, args.model, args.out, args.logit_threshold, args.threads)
