#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict

from common import ROOT, append_timeline, write_json


def ordered(rows):
    return sorted(rows, key=lambda row: (-float(row["confidence"]), int(row["class_id"]), float(row["x1"])))


def compare(reference, candidate):
    reference, candidate = ordered(reference), ordered(candidate)
    paired = min(len(reference), len(candidate))
    return {
        "paired": paired,
        "class_agreement": sum(reference[i]["class_id"] == candidate[i]["class_id"] for i in range(paired)),
        "max_confidence_difference": max((abs(float(reference[i]["confidence"]) - float(candidate[i]["confidence"])) for i in range(paired)), default=0.0),
        "max_box_difference_px": max((max(abs(float(reference[i][key]) - float(candidate[i][key])) for key in ("x1", "y1", "x2", "y2")) for i in range(paired)), default=0.0),
        "detection_count_delta": abs(len(reference) - len(candidate)),
    }


def main():
    append_timeline("fp32_correctness_summary_recovery", "START", "offline; no inference")
    prediction_path = ROOT / "validation/fp32_correctness_predictions.csv"
    log_path = ROOT / "logs/fp32_correctness.log"
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8")))
    grouped = defaultdict(lambda: defaultdict(list))
    image_paths = {}
    for row in rows:
        sample = int(row["sample_index"])
        grouped[sample][row["backend"]].append(row)
        image_paths[sample] = row["image_path"]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    passes = [int(value) for value in re.findall(r"Correctness (\d+)/20 pass=True", log)]
    failures = re.findall(r"Correctness \d+/20 pass=False", log)
    recovered_rows = []
    classes = set()
    for sample in sorted(grouped):
        backends = grouped[sample]
        source = backends.get("PyTorch FP32", [])
        onnx = backends.get("ONNX FP32", [])
        trt = backends.get("TensorRT FP32", [])
        classes.update(int(row["class_id"]) for row in trt)
        recovered_rows.append({
            "sample_index": sample,
            "image_path": image_paths[sample],
            "source_detection_count": len(source),
            "onnx_detection_count": len(onnx),
            "tensorrt_detection_count": len(trt),
            **{f"onnx_{key}": value for key, value in compare(source, onnx).items()},
            **{f"tensorrt_{key}": value for key, value in compare(source, trt).items()},
            "original_in_memory_gate_pass": sample in passes,
            "raw_onnx_trt_difference_values": "NA: passed original thresholds but numeric row was not serialized",
        })
    all_pass = passes == list(range(1, 21)) and not failures and sorted(grouped) == list(range(1, 21)) and all(
        set(grouped[sample]) == {"PyTorch FP32", "ONNX FP32", "TensorRT FP32"} for sample in grouped
    ) and len(classes) >= 2
    summary = {
        "status": "PASS" if all_pass else "FAIL",
        "sample_count": len(grouped),
        "source_pt_executions": 20,
        "onnxruntime_executions": 20,
        "tensorrt_executions": 20,
        "tensorrt_classes_observed": sorted(classes),
        "class_diversity_pass": len(classes) >= 2,
        "original_console_pass_count": len(passes),
        "original_console_fail_count": len(failures),
        "rows": recovered_rows,
        "thresholds": {"raw_mean_abs": 1e-3, "raw_max_abs": 0.1, "count_delta": 2, "confidence": 0.02, "box_px": 2.0},
        "recovery_provenance": {
            "method": "offline reconstruction from the original 20-execution prediction CSV and 20/20 pass console evidence",
            "no_inference_executed": True,
            "reason": "The original run completed all 20 passes, wrote predictions, then failed serializing numpy.bool_ into JSON.",
            "limitations": "Per-image raw ONNX-versus-TensorRT numeric differences and engine timing values were not serialized; their threshold pass state is retained by the original console evidence and is not guessed.",
            "prediction_evidence": str(prediction_path),
            "console_evidence": str(log_path),
        },
    }
    write_json(ROOT / "validation/fp32_correctness_summary.json", summary)
    append_timeline("fp32_correctness_summary_recovery", summary["status"], "offline reconstruction; zero inference executions")
    if not all_pass:
        raise RuntimeError("Offline correctness summary recovery failed")
    print(f"Recovered PASS summary from {len(passes)} original pass lines and {len(rows)} prediction rows; no inference executed.")


if __name__ == "__main__":
    main()
