#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

TRT_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
if TRT_SITE not in sys.path:
    sys.path.insert(0, TRT_SITE)
import tensorrt as trt

from common import ROOT, append_timeline, write_json


def main():
    append_timeline("tensorrt_parser", "START", "read-only parse")
    onnx_path = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    inputs = [{"name": network.get_input(i).name, "shape": list(network.get_input(i).shape), "dtype": str(network.get_input(i).dtype)} for i in range(network.num_inputs)]
    outputs = [{"name": network.get_output(i).name, "shape": list(network.get_output(i).shape), "dtype": str(network.get_output(i).dtype)} for i in range(network.num_outputs)]
    passed = bool(ok) and not errors and inputs == [{"name": "images", "shape": [1, 3, 640, 960], "dtype": "DataType.FLOAT"}]
    report = {
        "status": "PASS" if passed else "FAIL", "parser_return": bool(ok), "errors": errors,
        "warnings": "See logs/tensorrt_parser.log for TensorRT verbose output", "inputs": inputs,
        "outputs": outputs, "explicit_batch": True, "unsupported_operations": errors,
        "plugin_requirements": "none reported", "tensorrt_version": trt.__version__,
    }
    write_json(ROOT / "validation/tensorrt_parser_report.json", report)
    append_timeline("tensorrt_parser", report["status"], f"errors={len(errors)}")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("TensorRT parser validation failed")


if __name__ == "__main__":
    main()
