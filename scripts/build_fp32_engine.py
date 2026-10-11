#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
import time

# On this Jetson image, NumPy must initialize before the user-site TensorRT
# binding. The reverse import order terminates Python with SIGILL before main.
import numpy as np  # noqa: F401

TRT_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
if TRT_SITE not in sys.path:
    sys.path.insert(0, TRT_SITE)
import tensorrt as trt

from common import ROOT, append_timeline, file_metadata, record_command, write_json
from tensorrt_runtime import TensorRTRunner


def main():
    append_timeline("engine_build", "START", "strict FP32")
    onnx_path = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    engine_path = ROOT / "engine/oct_albument2_v6_tensorrt_fp32_1x3x640x960.engine"
    if engine_path.exists():
        raise FileExistsError(f"Fresh engine target already exists: {engine_path}")
    command = "TensorRT Python explicit-batch build: FP16=OFF INT8=OFF TF32=OFF DLA=OFF workspace=1024MiB"
    record_command(command)
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT parse failed during build: {errors}")
    config = builder.create_builder_config()
    config.max_workspace_size = 1024 * 1024 * 1024
    for flag_name in ("FP16", "INT8", "TF32", "GPU_FALLBACK"):
        flag = getattr(trt.BuilderFlag, flag_name, None)
        if flag is not None:
            config.clear_flag(flag)
    if hasattr(config, "default_device_type"):
        config.default_device_type = trt.DeviceType.GPU
    flags_before = int(config.flags)
    started_wall = time.time()
    started = time.perf_counter()
    engine = builder.build_engine(network, config)
    duration = time.perf_counter() - started
    if engine is None:
        raise RuntimeError("TensorRT build_engine returned None")
    engine_path.write_bytes(engine.serialize())
    ended_wall = time.time()
    build_config = {
        "precision": "strict FP32", "explicit_batch": True, "input_shape": [1, 3, 640, 960],
        "workspace_bytes": 1024 * 1024 * 1024, "workspace_mib": 1024,
        "builder_flags_integer": flags_before,
        "FP16_flag": "OFF", "INT8_flag": "OFF", "TF32_flag": "OFF", "GPU_FALLBACK_flag": "OFF",
        "DLA": "OFF; default device GPU", "calibration": "NONE", "mixed_precision": "OFF",
        "timing_cache": "fresh builder process; no historical timing cache loaded or saved",
        "python_import_order": "NumPy initialized before TensorRT to avoid a documented pre-main SIGILL on this Jetson image",
        "tactic_sources": "TensorRT 8.0.1 defaults", "builder_optimization_level": "API unavailable/not configured",
        "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
        "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
        "tensorrt_version": trt.__version__, "cuda_version": "10.2", "gpu_architecture": "NVIDIA Tegra X1 (Maxwell)",
        "command_description": command,
    }
    write_json(ROOT / "engine/engine_build_config.json", build_config)
    write_json(ROOT / "engine/engine_build_duration.json", {
        "start_epoch": started_wall, "end_epoch": ended_wall, "duration_s": duration,
    })
    with TensorRTRunner(engine_path) as runner:
        runtime_metadata = runner.metadata()
    metadata = {
        **file_metadata(engine_path), "build_config": build_config,
        "runtime_deserialization_validation": "PASS", **runtime_metadata,
        "no_inference_executed_during_metadata_validation": True,
    }
    write_json(ROOT / "engine/engine_metadata.json", metadata)
    (ROOT / "engine/engine_sha256.txt").write_text(f"{metadata['sha256']}  {engine_path.name}\n", encoding="utf-8")
    (ROOT / "engine/engine_build_log.txt").write_text(json.dumps({"status": "PASS", "config": build_config, "duration_s": duration, "engine": metadata}, indent=2) + "\n", encoding="utf-8")
    append_timeline("engine_build", "PASS", f"duration_s={duration:.3f}")
    print(json.dumps({"status": "PASS", "duration_s": duration, "engine": file_metadata(engine_path)}, indent=2))


if __name__ == "__main__":
    main()
