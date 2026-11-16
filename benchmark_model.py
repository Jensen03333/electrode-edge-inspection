"""Compare FP32 and INT8 segmentation on the same held-out frames using CPU ORT."""

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
from PIL import Image


def read_manifest(path, root, masks=False):
    """CSV paths are relative to root. All crops of one sample share a group."""
    root = root.resolve()
    required = {"file", "group"} | ({"mask"} if masks else set())
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    seen = set()
    for row in rows:
        row["group"] = row["group"].strip()
        if not row["group"]:
            raise ValueError("Every frame needs a source group")
        for key in required - {"group"}:
            relative = Path(row[key])
            target = (root / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"Invalid {key} path in {path}: {relative}")
            row[key] = target.relative_to(root).as_posix()
        if row["file"] in seen:
            raise ValueError(f"Duplicate frame: {row['file']}")
        seen.add(row["file"])
    return rows


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_session(model, threads, profile_prefix=None):
    if threads < 1:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def input_spec(session):
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected one image input and one binary segmentation output")
    node = inputs[0]
    if node.type != "tensor(float)" or len(node.shape) != 4:
        raise ValueError("Expected NCHW float32 input")
    batch, channels, height, width = node.shape
    if isinstance(batch, int) and batch != 1:
        raise ValueError(f"Input must accept batch size 1: {node.shape}")
    if channels not in (1, 3) or not all(isinstance(x, int) and x > 0 for x in (height, width)):
        raise ValueError(f"Expected one or three channels and fixed H/W: {node.shape}")
    return node.name, channels, (width, height)


def load_frame(path, spec):
    _, channels, size = spec
    with Image.open(path) as image:
        original_size = image.size
        image = image.convert("L" if channels == 1 else "RGB")
        image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    chw = array[None, :, :] if channels == 1 else array.transpose(2, 0, 1)
    return np.ascontiguousarray(chw[None]), original_size


def infer(session, spec, path, threshold):
    start = perf_counter_ns()
    tensor, original_size = load_frame(path, spec)
    output = session.run(None, {spec[0]: tensor})[0]
    if output.ndim != 4 or output.shape[:2] != (1, 1) or not np.isfinite(output).all():
        raise ValueError("Expected finite (1,1,H,W) crack logits")
    # Map logits back to the input canvas before applying the threshold.
    logits = Image.fromarray(output[0, 0]).resize(original_size, Image.Resampling.BILINEAR)
    prediction = np.asarray(logits) >= threshold
    return prediction, (perf_counter_ns() - start) / 1e6


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Expected a single-channel binary surface-crack mask: {path}")
    return mask != 0


def summarize(records):
    times = [r["service_ms"] for r in records]
    total = {key: sum(r[key] for r in records) for key in (
        "tp_px", "fp_px", "fn_px", "positive_frame", "missed_positive_frame", "false_alarm_frame"
    )}
    tp, fp, fn = (total[k] for k in ("tp_px", "fp_px", "fn_px"))
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    return {
        **total, "images": len(records),
        "dice_micro": ratio(2 * tp, 2 * tp + fp + fn),
        "pixel_recall": ratio(tp, tp + fn),
        "negative_frame": len(records) - total["positive_frame"],
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
    }


def benchmark(args):
    rows = read_manifest(args.evaluation, args.data_root, masks=True)
    frame_root = Path(os.path.commonpath([(args.data_root / r["file"]).resolve().parent for r in rows]))
    report_path = args.int8.with_suffix(".calibration.json")
    calibration = json.loads(report_path.read_text(encoding="utf-8"))
    if file_hash(args.fp32) != calibration["fp32_sha256"] or file_hash(args.int8) != calibration["int8_sha256"]:
        raise ValueError("Model files differ from the calibration report")
    forbidden_groups = set(calibration["calibration_groups"])
    forbidden_hashes = set(calibration["calibration_frame_sha256"])
    for row in rows:
        if row["group"] in forbidden_groups or file_hash(args.data_root / row["file"]) in forbidden_hashes:
            raise ValueError(f"Evaluation frame overlaps calibration: {row['file']}")
    if args.warmup < 0 or not np.isfinite(args.logit_threshold):
        raise ValueError("Invalid warmup or logit threshold")
    sessions = {k: cpu_session(p, args.threads) for k, p in (("fp32", args.fp32), ("int8", args.int8))}
    specs = {k: input_spec(session) for k, session in sessions.items()}
    if specs["fp32"][1:] != specs["int8"][1:]:
        raise ValueError("Input geometries differ")
    for k, session in sessions.items():
        for i in range(args.warmup):
            infer(session, specs[k], args.data_root / rows[i % len(rows)]["file"], args.logit_threshold)
    records = {"fp32": [], "int8": []}
    changed_pixels = evaluated_pixels = lost_crack_frames = recovered_crack_frames = 0
    for i, row in enumerate(rows):
        truth = read_mask(args.data_root / row["mask"])
        predictions = {}
        for key in (("fp32", "int8") if i % 2 == 0 else ("int8", "fp32")):
            pred, ms = infer(sessions[key], specs[key], args.data_root / row["file"], args.logit_threshold)
            predictions[key] = pred
            if pred.shape != truth.shape:
                raise ValueError(f"Image/mask geometry differs: {row['file']}")
            tp = int(np.count_nonzero(pred & truth))
            positive = bool(truth.any())
            records[key].append({
                "file": (args.data_root / row["file"]).resolve().relative_to(frame_root).as_posix(),
                "service_ms": ms, "dataset_file": row["file"],
                "tp_px": tp, "fp_px": int(pred.sum()) - tp, "fn_px": int(truth.sum()) - tp,
                "positive_frame": int(positive), "missed_positive_frame": int(positive and tp == 0),
                "false_alarm_frame": int(not positive and bool(pred.any())),
            })
        changed_pixels += int(np.count_nonzero(predictions["int8"] != predictions["fp32"]))
        evaluated_pixels += truth.size
        fp32_hit = records["fp32"][-1]["tp_px"] > 0
        int8_hit = records["int8"][-1]["tp_px"] > 0
        lost_crack_frames += int(fp32_hit and not int8_hit)
        recovered_crack_frames += int(int8_hit and not fp32_hit)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "cpu": platform.processor(), "architecture": platform.machine(),
        "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
        "threads": args.threads, "warmup_frames_per_model": args.warmup,
        "logit_threshold": args.logit_threshold, "evaluation_manifest_sha256": file_hash(args.evaluation),
        "replay_image_root_relative_to_data_root": frame_root.relative_to(args.data_root.resolve()).as_posix(),
        "timed_scope": "file read + bilinear resize + [0,1] conversion + inference + mask decision",
        "missed_positive_frame": "positive frame with zero overlapping predicted crack pixels",
        "false_alarm_frame": "frame without a surface crack with any predicted crack pixel",
        "int8_vs_fp32": {"pixel_decision_change_fraction": changed_pixels / evaluated_pixels,
                         "positive_frames_losing_all_overlap": lost_crack_frames,
                         "positive_frames_gaining_overlap": recovered_crack_frames},
    }
    for key, values in records.items():
        with (args.output / f"{key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        model_path = getattr(args, key)
        summary[key] = {**summarize(values), "onnx_bytes": model_path.stat().st_size, "sha256": file_hash(model_path)}
    content = json.dumps(summary, indent=2)
    (args.output / "comparison.json").write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fp32", "int8", "evaluation", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    benchmark(parser.parse_args())
