"""Run the two frozen models on attributed coating crack photographs from a pilot line."""

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_model import cpu_session, file_hash, infer, input_spec


ROOT = Path(__file__).resolve().parent
ARTICLE = "https://doi.org/10.3390/batteries9020111"
SOURCE = (
    "https://pub.mdpi-res.com/batteries/batteries-09-00111/article_deploy/"
    "html/images/batteries-09-00111-g003.png"
)
FIGURE_SHA256 = "79a7c5065467a4a564d156db0e66d59e4e53c2fffc34471565198b68de7d714e"
CROPS = {
    "3b-coating-crack": (1120, 2260, 1760, 2740),
    "3e-mud-crack": (1005, 185, 1425, 500),
}


def open_figure(local_path):
    if local_path:
        payload = local_path.read_bytes()
    else:
        request = Request(SOURCE, headers={"User-Agent": "ElectrodeEdgeInspection/1.0 (figure attribution)"})
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != FIGURE_SHA256:
        raise ValueError("The original figure changed; the crop coordinates must be checked visually.")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def annotate(crop, predicted):
    pixels = np.array(crop)
    pixels[predicted] = np.rint(0.35 * pixels[predicted] + 0.65 * np.array([238, 34, 117]))
    return Image.fromarray(pixels)


def make_row(sample, figure_name, overlays, target):
    tile = (320, 240)
    panel = Image.new("RGB", (960, 301), "white")
    draw = ImageDraw.Draw(panel)
    try:
        title_font = ImageFont.load_default(size=17)
        foot_font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow 10.0 has a fixed-size default bitmap font.
        title_font = foot_font = ImageFont.load_default()
    for i, (label, image) in enumerate(zip(("Source " + figure_name, "FP32 prediction", "INT8 prediction"),
                                           (sample, *overlays))):
        draw.text((i * tile[0] + 6, 4), label, fill="black", font=title_font)
        panel.paste(image.resize(tile, Image.Resampling.LANCZOS), (i * tile[0], 25))
    draw.text((5, 270), "Figure 3 crop: Schoo, Moschner, Huelsmann & Kwade, Batteries (2023), CC BY 4.0.",
              fill="black", font=foot_font)
    draw.text((5, 285), "Magenta shows model output. Reference pixel masks are unavailable.",
              fill="black", font=foot_font)
    panel.save(target, format="PNG", optimize=True)


def run(args):
    source = open_figure(args.figure)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    threshold = benchmark["logit_threshold"]
    sessions = {}
    for label, path in (("fp32", args.fp32), ("int8", args.int8)):
        if file_hash(path) != benchmark[label]["sha256"]:
            raise ValueError(f"{label}: model fingerprint differs from the frozen benchmark.")
        session = cpu_session(path, threads=1)
        sessions[label] = (session, input_spec(session))
    args.out.mkdir(parents=True, exist_ok=True)
    samples = {}
    with tempfile.TemporaryDirectory() as work:
        for name, coords in CROPS.items():
            patch = source.crop(coords)
            path = Path(work) / f"{name}.png"
            patch.save(path)
            outputs = {}
            for label, (session, spec) in sessions.items():
                outputs[label], _ = infer(session, spec, path, threshold)
            make_row(patch, name[:2], [annotate(patch, out) for out in outputs.values()], args.out / f"{name}.png")
            samples[name] = {
                "crop_xyxy_exclusive": coords,
                "width": patch.width,
                "height": patch.height,
                "fp32_predicted_pixels": int(outputs["fp32"].sum()),
                "int8_predicted_pixels": int(outputs["int8"].sum()),
                "different_prediction_pixels": int(np.count_nonzero(outputs["fp32"] != outputs["int8"])),
            }
    summary = {
        "article": ARTICLE,
        "source_figure": SOURCE,
        "source_figure_size": source.size,
        "source_figure_sha256": FIGURE_SHA256,
        "source_license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "model_sha256": {label: benchmark[label]["sha256"] for label in sessions},
        "logit_threshold": threshold,
        "evidence": "Visual examples only; publication panels have no pixel annotations.",
        "samples": samples,
    }
    (args.out / "external.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Use a local copy of the same unmodified publication figure.")
    parser.add_argument("--fp32", type=Path, default=ROOT / "models/fp32.onnx")
    parser.add_argument("--int8", type=Path, default=ROOT / "models/int8.onnx")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "reports/benchmark.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/external")
    run(parser.parse_args())
