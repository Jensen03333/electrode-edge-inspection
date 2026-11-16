"""CPU pipeline regression on a fixed one-convolution graph, independent of learned weights."""

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from benchmark_model import benchmark
from quantize_model import quantize


class QuantizationPipelineTest(unittest.TestCase):
    def test_quantize_compare_and_reject_reused_group(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = [1, 1, 8, 8]
            node = helper.make_node("Conv", ["frame", "weights", "bias"], ["logits"], name="binary_conv")
            initializers = [numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weights"),
                            numpy_helper.from_array(np.array([-0.5], np.float32), "bias")]
            graph = helper.make_graph(
                [node], "regression", [helper.make_tensor_value_info("frame", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("logits", TensorProto.FLOAT, shape)], initializers,
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
            fp32 = root / "float.onnx"
            int8 = root / "int8.onnx"
            onnx.save(model, str(fp32))
            for index in (1, 2, 3):
                frame = np.zeros((8, 8), np.uint8)
                frame[index, :] = 255
                Image.fromarray(frame).save(root / f"{index}.png")
            calibration = root / "cal.csv"
            evaluation = root / "test.csv"
            calibration.write_text("file,group\n1.png,roll-A\n", encoding="utf-8")
            evaluation.write_text("file,mask,group\n2.png,2.png,roll-B\n3.png,3.png,roll-C\n", encoding="utf-8")
            quant_args = argparse.Namespace(fp32=fp32, output=int8, calibration=calibration, evaluation=evaluation, data_root=root)
            quantize(quant_args)
            bench_args = argparse.Namespace(fp32=fp32, int8=int8, evaluation=evaluation, data_root=root,
                                           output=root / "out", threads=1, warmup=1, logit_threshold=0.0)
            benchmark(bench_args)
            summary = json.loads((root / "out" / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["int8"]["dice_micro"], 1)
            self.assertEqual(summary["fp32"]["images"], 2)
            with (root / "out" / "int8.csv").open(newline="", encoding="utf-8") as stream:
                self.assertTrue(all(float(row["service_ms"]) > 0 for row in csv.DictReader(stream)))
            evaluation.write_text("file,mask,group\n2.png,2.png,roll-A\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps calibration"):
                benchmark(bench_args)
            with self.assertRaisesRegex(ValueError, "share an image or source group"):
                quantize(quant_args)


if __name__ == "__main__":
    unittest.main()
