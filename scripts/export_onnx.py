#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from common import HEIGHT, ROOT, WIDTH, append_timeline, file_metadata, record_command, write_json


def main():
    source_copy = ROOT / "model/oct_albument2_v6_source.pt"
    target = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    if target.exists():
        raise FileExistsError(f"Fresh export target already exists: {target}")
    command = (
        f"YOLO({source_copy}).export(format='onnx', imgsz=({HEIGHT},{WIDTH}), batch=1, "
        "opset=12, dynamic=False, simplify=False, half=False, int8=False, nms=False, device=0)"
    )
    (ROOT / "onnx/onnx_export_command.txt").write_text(command + "\n", encoding="utf-8")
    record_command(command)
    append_timeline("onnx_export", "START", command)
    started = time.perf_counter()
    model = YOLO(str(source_copy))
    exported = Path(model.export(
        format="onnx", imgsz=(HEIGHT, WIDTH), batch=1, opset=12, dynamic=False,
        simplify=False, half=False, int8=False, nms=False, device=0, verbose=True,
    ))
    duration = time.perf_counter() - started
    if not exported.exists():
        raise RuntimeError(f"Ultralytics reported export path does not exist: {exported}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(target))
    write_json(ROOT / "onnx/onnx_export_result.json", {
        "status": "PASS", "duration_s": duration, "command": command,
        "source_model_copy": file_metadata(source_copy), "onnx": file_metadata(target),
        "precision": "FP32", "static_shape": [1, 3, HEIGHT, WIDTH], "opset_requested": 12,
        "internal_export_model_executions": "Ultralytics export performs implementation-internal tracing forwards; outside correctness/warmup/official execution counts.",
    })
    append_timeline("onnx_export", "PASS", f"duration_s={duration:.3f}")
    print(json.dumps(file_metadata(target), indent=2))


if __name__ == "__main__":
    main()
