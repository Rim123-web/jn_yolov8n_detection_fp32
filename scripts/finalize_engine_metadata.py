#!/usr/bin/env python3
from __future__ import annotations

import json

from common import ROOT, append_timeline, file_metadata, read_json, write_json
from tensorrt_runtime import TensorRTRunner


def main():
    append_timeline("engine_postbuild_validation", "START", "separate clean process; no inference")
    engine_path = ROOT / "engine/oct_albument2_v6_tensorrt_fp32_1x3x640x960.engine"
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Fresh engine missing: {engine_path}")
    build_config = read_json(ROOT / "engine/engine_build_config.json", {})
    required = {
        "precision": "strict FP32",
        "FP16_flag": "OFF",
        "INT8_flag": "OFF",
        "TF32_flag": "OFF",
        "DLA": "OFF; default device GPU",
        "calibration": "NONE",
        "mixed_precision": "OFF",
    }
    mismatches = {key: {"expected": value, "actual": build_config.get(key)} for key, value in required.items() if build_config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Strict-FP32 build evidence mismatch: {mismatches}")
    with TensorRTRunner(engine_path) as runner:
        runtime_metadata = runner.metadata()
    bindings = runtime_metadata["binding_metadata"]
    expected_input = [row for row in bindings if row["is_input"]]
    if len(expected_input) != 1 or expected_input[0]["runtime_shape"] != [1, 3, 640, 960] or expected_input[0]["dtype"] != "DataType.FLOAT":
        raise RuntimeError(f"Engine input validation failed: {expected_input}")
    outputs = [row for row in bindings if not row["is_input"]]
    if not outputs or any(row["dtype"] != "DataType.FLOAT" for row in outputs):
        raise RuntimeError(f"Engine output validation failed: {outputs}")
    metadata = {
        **file_metadata(engine_path),
        "status": "PASS",
        "build_config": build_config,
        "runtime_deserialization_validation": "PASS",
        **runtime_metadata,
        "no_inference_executed_during_metadata_validation": True,
        "recovery_note": "Fresh engine serialized by the original builder process; separate metadata validation completed after replacing TensorRT 8.0.1 trt.nptype with an explicit NumPy 1.24-compatible dtype mapping. Engine was not rebuilt.",
    }
    write_json(ROOT / "engine/engine_metadata.json", metadata)
    (ROOT / "engine/engine_sha256.txt").write_text(f"{metadata['sha256']}  {engine_path.name}\n", encoding="utf-8")
    (ROOT / "engine/engine_build_log.txt").write_text(json.dumps({
        "status": "PASS",
        "build": "fresh engine serialized successfully",
        "postbuild_validation": "PASS in a separate clean process",
        "config": build_config,
        "duration": read_json(ROOT / "engine/engine_build_duration.json", {}),
        "engine": metadata,
    }, indent=2) + "\n", encoding="utf-8")
    append_timeline("engine_postbuild_validation", "PASS", "deserialized; static FP32 bindings; no inference")
    print(json.dumps({"status": "PASS", "engine": file_metadata(engine_path), "bindings": bindings}, indent=2))


if __name__ == "__main__":
    main()
