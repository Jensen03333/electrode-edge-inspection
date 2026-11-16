"""Calibrate convolution weights and activations to CPU INT8 QDQ on labelled source groups."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark_model import cpu_session, file_hash, input_spec, load_frame, read_manifest


def audit_convolutions(model):
    import onnx

    graph = onnx.load(str(model))
    onnx.checker.check_model(graph)
    nodes = {name: node for node in graph.graph.node for name in node.output}
    tensors = {tensor.name: tensor for tensor in graph.graph.initializer}
    convs = [node for node in graph.graph.node if node.op_type == "Conv"]
    quantized = 0
    for conv in convs:
        activation = nodes.get(conv.input[0])
        weight = nodes.get(conv.input[1])
        if not activation or not weight or activation.op_type != "DequantizeLinear" or weight.op_type != "DequantizeLinear":
            continue
        a_quant = nodes.get(activation.input[0])
        w_tensor = tensors.get(weight.input[0])
        if a_quant and a_quant.op_type == "QuantizeLinear" and w_tensor and w_tensor.data_type == onnx.TensorProto.INT8:
            quantized += 1
    if not quantized:
        raise ValueError("No convolution with activation QDQ and stored INT8 weights")
    return {"conv_nodes": len(convs), "conv_nodes_with_int8_weights_and_activation_qdq": quantized}


def quantize(args):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    rows = read_manifest(args.calibration, args.data_root)
    held_out = read_manifest(args.evaluation, args.data_root, masks=True)
    groups = {r["group"] for r in rows}
    hashes = {file_hash(args.data_root / r["file"]) for r in rows}
    if any(r["group"] in groups or file_hash(args.data_root / r["file"]) in hashes for r in held_out):
        raise ValueError("Calibration and evaluation share an image or source group")
    if args.fp32.resolve() == args.output.resolve():
        raise ValueError("FP32 input and INT8 output must have distinct paths")
    spec = input_spec(cpu_session(args.fp32, 1))

    class Frames(CalibrationDataReader):
        def __init__(self):
            self.items = iter(rows)

        def get_next(self):
            row = next(self.items, None)
            if row is None:
                return None
            tensor, _ = load_frame(args.data_root / row["file"], spec)
            return {spec[0]: tensor}

        def rewind(self):
            self.items = iter(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        candidate = Path(temp) / "model.onnx"
        quantize_static(
            str(args.fp32), str(candidate), Frames(),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Conv"],
        )
        audit = audit_convolutions(candidate)
        # Check the CPU provider can execute the exported graph before publishing it.
        session = cpu_session(candidate, 1, profile_prefix=Path(temp) / "cpu_profile")
        first = Frames().get_next()
        session.run(None, first)
        profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        cpu_kernels = Counter(event["args"]["op_name"] for event in profile
                              if event.get("args", {}).get("provider") == "CPUExecutionProvider"
                              and "op_name" in event["args"])
        del session
        report = {
            **audit, "calibration_frames": len(rows),
            "calibration_groups": sorted(groups), "calibration_frame_sha256": sorted(hashes),
            "calibration_manifest_sha256": file_hash(args.calibration),
            "held_out_manifest_sha256": file_hash(args.evaluation),
            "fp32_sha256": file_hash(args.fp32), "int8_sha256": file_hash(candidate),
            "weight_format": "per-channel signed INT8", "activation_format": "static signed INT8 QDQ",
            "target_operators": ["Conv"], "cpu_profile_one_frame_kernels": dict(cpu_kernels),
        }
        sidecar = Path(temp) / "calibration.json"
        sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
        candidate.replace(args.output)
        sidecar.replace(args.output.with_suffix(".calibration.json"))
    print(f"Quantized {audit['conv_nodes_with_int8_weights_and_activation_qdq']}/{audit['conv_nodes']} convolutions "
          f"using {len(rows)} frames -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fp32", "calibration", "evaluation", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    quantize(parser.parse_args())
