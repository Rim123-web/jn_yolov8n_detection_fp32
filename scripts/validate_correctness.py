#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from common import CLASS_NAMES, ROOT, SEED, append_timeline, collect_images, write_csv, write_json
from postprocessing import postprocess_yolov8
from preprocessing import preprocess_image
from tensorrt_runtime import TensorRTRunner


def torch_output(model, array):
    tensor = torch.from_numpy(array).to("cuda:0", dtype=torch.float32)
    torch.cuda.synchronize()
    with torch.inference_mode():
        output = model(tensor)
    torch.cuda.synchronize()
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().numpy()


def sorted_detections(detections):
    return sorted(detections, key=lambda x: (-x["confidence"], x["class_id"], x["box"][0]))


def comparison(reference, candidate):
    reference = sorted_detections(reference)
    candidate = sorted_detections(candidate)
    count = min(len(reference), len(candidate))
    class_agreement = sum(reference[i]["class_id"] == candidate[i]["class_id"] for i in range(count))
    confidence_difference = max((abs(reference[i]["confidence"] - candidate[i]["confidence"]) for i in range(count)), default=0.0)
    box_difference = max((max(abs(a - b) for a, b in zip(reference[i]["box"], candidate[i]["box"])) for i in range(count)), default=0.0)
    return {
        "paired": count,
        "class_agreement": class_agreement,
        "max_confidence_difference": confidence_difference,
        "max_box_difference_px": box_difference,
        "detection_count_delta": abs(len(reference) - len(candidate)),
    }


def valid_detections(detections, meta):
    for detection in detections:
        cls = detection["class_id"]
        conf = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]
        if not 0 <= cls < len(CLASS_NAMES) or not np.isfinite(conf) or not 0 <= conf <= 1:
            return False
        if not all(np.isfinite(x) for x in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
            return False
        if x1 < -1e-4 or y1 < -1e-4 or x2 > meta["original_width"] + 1e-4 or y2 > meta["original_height"] + 1e-4:
            return False
    return True


def main():
    append_timeline("fp32_correctness", "START", "exactly 20 images")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = YOLO(str(ROOT / "model/oct_albument2_v6_source.pt")).model.to("cuda:0").float().eval()
    session = ort.InferenceSession(str(ROOT / "onnx/oct_albument2_v6_inference_only.onnx"), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows = []
    prediction_rows = []
    class_ids = set()
    engine = ROOT / "engine/oct_albument2_v6_tensorrt_fp32_1x3x640x960.engine"
    with TensorRTRunner(engine) as runner:
        for index, image_path in enumerate(collect_images()[:20], 1):
            array, meta, _ = preprocess_image(image_path)
            source_raw = torch_output(model, array)
            onnx_raw = session.run([output_name], {input_name: array})[0]
            trt_outputs, trt_timing = runner.run_timed(array)
            trt_raw = trt_outputs[0]
            source_dets, _ = postprocess_yolov8(source_raw, meta)
            onnx_dets, _ = postprocess_yolov8(onnx_raw, meta)
            trt_dets, _ = postprocess_yolov8(trt_raw, meta)
            raw_diff = np.abs(onnx_raw.astype(np.float32) - trt_raw.astype(np.float32))
            onnx_compare = comparison(source_dets, onnx_dets)
            trt_compare = comparison(source_dets, trt_dets)
            finite = np.isfinite(source_raw).all() and np.isfinite(onnx_raw).all() and np.isfinite(trt_raw).all()
            valid = valid_detections(source_dets, meta) and valid_detections(onnx_dets, meta) and valid_detections(trt_dets, meta)
            passed = (
                finite and valid and source_raw.shape == onnx_raw.shape == trt_raw.shape
                and float(raw_diff.mean()) <= 1e-3 and float(raw_diff.max()) <= 0.1
                and trt_compare["detection_count_delta"] <= 2
                and trt_compare["class_agreement"] == trt_compare["paired"]
                and trt_compare["max_confidence_difference"] <= 0.02
                and trt_compare["max_box_difference_px"] <= 2.0
            )
            for backend, detections in (("PyTorch FP32", source_dets), ("ONNX FP32", onnx_dets), ("TensorRT FP32", trt_dets)):
                for det_index, detection in enumerate(detections):
                    prediction_rows.append({
                        "sample_index": index, "image_path": str(image_path), "backend": backend,
                        "detection_index": det_index, "class_id": detection["class_id"],
                        "class_name": detection["class_name"], "confidence": detection["confidence"],
                        "x1": detection["box"][0], "y1": detection["box"][1],
                        "x2": detection["box"][2], "y2": detection["box"][3],
                    })
                    if backend == "TensorRT FP32":
                        class_ids.add(detection["class_id"])
            rows.append({
                "sample_index": index, "image_path": str(image_path), "output_shape": str(tuple(trt_raw.shape)),
                "all_outputs_finite": finite, "detections_valid": valid,
                "raw_onnx_trt_max_abs_difference": float(raw_diff.max()),
                "raw_onnx_trt_mean_abs_difference": float(raw_diff.mean()),
                "source_detection_count": len(source_dets), "onnx_detection_count": len(onnx_dets),
                "tensorrt_detection_count": len(trt_dets), **{f"onnx_{k}": v for k, v in onnx_compare.items()},
                **{f"tensorrt_{k}": v for k, v in trt_compare.items()},
                "tensorrt_engine_ms": trt_timing["tensorrt_engine_ms"], "pass": passed,
            })
            print(f"Correctness {index}/20 pass={passed}", flush=True)
    write_csv(ROOT / "validation/fp32_correctness_predictions.csv", prediction_rows)
    all_pass = len(rows) == 20 and all(row["pass"] for row in rows) and len(class_ids) >= 2
    summary = {
        "status": "PASS" if all_pass else "FAIL", "sample_count": len(rows),
        "source_pt_executions": 20, "onnxruntime_executions": 20, "tensorrt_executions": 20,
        "tensorrt_classes_observed": sorted(class_ids), "class_diversity_pass": len(class_ids) >= 2,
        "rows": rows, "thresholds": {"raw_mean_abs": 1e-3, "raw_max_abs": 0.1, "count_delta": 2, "confidence": 0.02, "box_px": 2.0},
    }
    write_json(ROOT / "validation/fp32_correctness_summary.json", summary)
    append_timeline("fp32_correctness", summary["status"], "20 TensorRT executions recorded separately")
    if not all_pass:
        raise RuntimeError("20-image correctness gate failed")


if __name__ == "__main__":
    main()
