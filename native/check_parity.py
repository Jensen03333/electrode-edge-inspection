"""Compare native CPU inference against Python ORT on identical real grayscale bytes."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("exe", "dll", "frame", "fp32", "int8", "report"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.frame) as image:
        frame = image.convert("L").resize((320, 240), Image.Resampling.BILINEAR)
    tensor = (np.asarray(frame, dtype=np.float32) / 255.0)[None, None]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    report = {
        "frame": args.frame.name,
        "frame_sha256": hashlib.sha256(args.frame.read_bytes()).hexdigest(),
        "input_shape": list(tensor.shape), "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider", "threads": 1,
        "scope": "Python decodes and resizes one held-out photograph. C++ reads identical PGM bytes and runs ORT.",
        "logit_atol": 1e-5, "logit_rtol": 1e-5, "decision_threshold": 0.0,
    }
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        pgm = temporary / "frame.pgm"
        frame.save(pgm)
        for label, model in (("fp32", args.fp32), ("int8", args.int8)):
            output = temporary / f"{label}.f32"
            subprocess.run([str(path.resolve()) for path in (args.exe, args.dll, model, pgm, output)], check=True)
            native = np.fromfile(output, dtype="<f4").reshape(tensor.shape)
            session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
            reference = session.run(["logits"], {"image": tensor})[0]
            if not (np.isfinite(reference).all() and np.isfinite(native).all()):
                raise ValueError(f"Non-finite logits for {label}")
            different = int(np.count_nonzero((reference >= 0) != (native >= 0)))
            np.testing.assert_allclose(native, reference, rtol=1e-5, atol=1e-5)
            if different:
                raise ValueError(f"{different} pixel decisions differ for {label}")
            report[label] = {
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "compared_pixels": native.size, "different_decisions": different,
                "max_abs_logit_error": float(np.max(np.abs(native - reference))),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
