#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from common import CLASS_NAMES, ROOT, SOURCE_PT, finite_nonnegative, read_json, sha256_file, write_json
from create_manifest import verify as verify_manifest


REQUIRED_SCRIPTS = [
    "export_onnx.py", "validate_onnx.py", "validate_tensorrt_parser.py", "build_fp32_engine.py",
    "preprocessing.py", "tensorrt_runtime.py", "postprocessing.py", "nms.py",
    "validate_correctness.py", "run_jtop_health_test.py", "capture_idle.py",
    "benchmark_fp32_detection.py", "calculate_detection_metrics.py", "analyze_timings.py",
    "analyze_energy.py", "analyze_resources.py", "generate_figures.py", "generate_reports.py",
    "validate_package.py", "create_manifest.py", "package_results.sh",
]
REQUIRED_REPORTS = [
    "full_benchmark_report.md", "full_benchmark_report.html", "methodology.md", "results.md",
    "limitations.md", "reproducibility.md", "article_ready_results.md",
    "defense_ready_summary.md", "failure_and_recovery_log.md",
]
TIMING_FIELDS = [
    "file_access_decode_ms", "exif_orientation_ms", "color_conversion_ms", "resize_calculation_ms",
    "image_resize_ms", "padding_letterbox_ms", "pixel_scaling_ms", "layout_conversion_ms",
    "fp32_contiguous_ms", "preprocessing_total_ms", "h2d_ms", "tensorrt_engine_ms", "d2h_ms",
    "transfer_inclusive_inference_ms", "raw_output_decode_ms", "box_conversion_ms",
    "confidence_calculation_ms", "confidence_filtering_ms", "class_selection_ms",
    "nms_preparation_ms", "nms_ms", "coordinate_rescaling_ms", "box_clipping_ms",
    "result_formatting_ms", "postprocessing_total_ms", "end_to_end_ms",
]


def check(name, condition, evidence=""):
    return {"gate": name, "pass": bool(condition), "evidence": evidence}


def main():
    gates = []
    model_meta = read_json(ROOT / "model/source_model_metadata.json", {})
    onnx_meta = read_json(ROOT / "onnx/onnx_metadata.json", {})
    engine_meta = read_json(ROOT / "engine/engine_metadata.json", {})
    engine_config = read_json(ROOT / "engine/engine_build_config.json", {})
    source_copy = ROOT / "model/oct_albument2_v6_source.pt"
    onnx_path = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    engine_path = ROOT / "engine/oct_albument2_v6_tensorrt_fp32_1x3x640x960.engine"
    gates += [
        check("source_pt_exists", source_copy.is_file(), str(source_copy)),
        check("source_hash_matches_original", source_copy.is_file() and SOURCE_PT.is_file() and sha256_file(source_copy) == sha256_file(SOURCE_PT) == model_meta.get("sha256"), model_meta.get("sha256", "")),
        check("onnx_exists_and_hash_matches", onnx_path.is_file() and sha256_file(onnx_path) == onnx_meta.get("sha256"), onnx_meta.get("sha256", "")),
        check("engine_exists_and_hash_matches", engine_path.is_file() and sha256_file(engine_path) == engine_meta.get("sha256"), engine_meta.get("sha256", "")),
        check("strict_fp32", all(engine_config.get(x) == "OFF" for x in ("FP16_flag", "INT8_flag", "TF32_flag")) and engine_config.get("calibration") == "NONE"),
        check("input_shape", onnx_meta.get("inputs", [{}])[0].get("shape") == [1, 3, 640, 960]),
        check("batch_one", onnx_meta.get("inputs", [{}])[0].get("shape", [None])[0] == 1),
        check("class_mapping", model_meta.get("class_names") == CLASS_NAMES),
    ]
    dataset_rows = list(csv.DictReader((ROOT / "dataset/test_image_manifest.csv").open(encoding="utf-8")))
    gates.append(check("dataset_manifest", len(dataset_rows) == 192 and len({r["image_path"] for r in dataset_rows}) == 192, f"rows={len(dataset_rows)}"))
    warmup_rows = list(csv.DictReader((ROOT / "raw/warmup_timings.csv").open(encoding="utf-8")))
    warmup_prov = read_json(ROOT / "raw/warmup_execution_provenance.json", {})
    gates += [
        check("warmup_rows_exactly_10", len(warmup_rows) == 10),
        check("warmup_paths_unique", len({r["image_path"] for r in warmup_rows}) == 10),
        check("warmup_engine_executions_10", warmup_prov.get("engine_executions") == 10),
        check("warmups_excluded", warmup_prov.get("excluded_from_official_latency_fps_metrics_resources_energy") is True),
    ]
    measured = list(csv.DictReader((ROOT / "raw/per_image_timings.csv").open(encoding="utf-8")))
    prediction_lines = (ROOT / "raw/predictions.jsonl").read_text(encoding="utf-8").splitlines()
    execution = read_json(ROOT / "raw/execution_count_provenance.json", {})
    gates += [
        check("measured_rows_exact", len(measured) == 192),
        check("prediction_image_rows_exact", len(prediction_lines) == 192),
        check("measured_paths_unique", len({r["image_path"] for r in measured}) == 192),
        check("failed_images_zero", execution.get("failed_measured_images") == 0),
        check("discarded_rows_zero", execution.get("discarded_measured_rows") == 0),
        check("hidden_executions_zero", execution.get("hidden_executions") == 0),
        check("official_execution_count", execution.get("accepted_total_benchmark_executions") == 202),
        check("all_timings_finite_nonnegative", all(finite_nonnegative(row[field]) for row in measured for field in TIMING_FIELDS)),
        check("independent_nms_timing", "nms_ms" in measured[0] and all(finite_nonnegative(r["nms_ms"]) for r in measured)),
        check("independent_engine_timing", "tensorrt_engine_ms" in measured[0]),
        check("end_to_end_timing", "end_to_end_ms" in measured[0]),
        check(
            "timing_units_and_decomposition",
            all(
                float(row["preprocessing_total_ms"]) < 2000.0
                and abs(
                    float(row["end_to_end_ms"])
                    - float(row["preprocessing_total_ms"])
                    - float(row["transfer_inclusive_inference_ms"])
                    - float(row["postprocessing_total_ms"])
                ) < 10.0
                for row in measured
            ),
            "preprocess <2000 ms and |end-to-end - preprocess - transfer-inclusive - postprocess| <10 ms for every row",
        ),
    ]
    gates += [
        check("onnx_checker", read_json(ROOT / "validation/onnx_checker_report.json", {}).get("status") == "PASS"),
        check("onnx_equivalence", read_json(ROOT / "validation/onnx_equivalence_summary.json", {}).get("status") == "PASS"),
        check("tensorrt_parser", read_json(ROOT / "validation/tensorrt_parser_report.json", {}).get("status") == "PASS"),
        check("correctness", read_json(ROOT / "validation/fp32_correctness_summary.json", {}).get("status") == "PASS"),
        check("jtop_health", read_json(ROOT / "monitoring/jtop_health_summary.json", {}).get("status") == "PASS"),
        check("idle_baseline", read_json(ROOT / "monitoring/idle_summary.json", {}).get("status") == "PASS"),
        check("energy_integration", read_json(ROOT / "energy/energy_summary.json", {}).get("status") == "PASS"),
        check("metrics", read_json(ROOT / "metrics/detection_metrics.json", {}).get("status") == "PASS"),
        check("telemetry_coverage", len(list(csv.DictReader((ROOT / "monitoring/active_samples.csv").open(encoding="utf-8")))) >= 2),
    ]
    figure_counts = {ext: len(list((ROOT / f"figures/{ext}").glob(f"*.{ext}"))) for ext in ("png", "svg", "pdf")}
    gates += [
        check("figures_all_formats", all(value == 33 for value in figure_counts.values()), str(figure_counts)),
        check("figure_source_data", len(list((ROOT / "figures/data").glob("*.csv"))) == 33),
        check("required_scripts", all((ROOT / "scripts" / name).is_file() for name in REQUIRED_SCRIPTS)),
        check("required_reports", all((ROOT / "reports" / name).is_file() for name in REQUIRED_REPORTS)),
        check("timing_unit_repair_evidence", read_json(ROOT / "validation/timing_unit_repair.json", {}).get("status") == "PASS"),
        check("atomic_bottleneck_definition", read_json(ROOT / "analysis/bottleneck_summary.json", {}).get("largest_mean_latency_stage") == "tensorrt_engine_ms"),
    ]
    manifest_rows, manifest_failures = verify_manifest()
    gates.append(check("hash_manifest", not manifest_failures, f"entries={len(manifest_rows)} failures={manifest_failures}"))
    status = "PASS" if all(gate["pass"] for gate in gates) else "FAIL"
    report = {
        "status": status, "gate_count": len(gates), "passed_gate_count": sum(g["pass"] for g in gates),
        "failed_gates": [g["gate"] for g in gates if not g["pass"]], "gates": gates,
        "manifest_entries_verified": len(manifest_rows), "manifest_failures": manifest_failures,
        "note": "Final validation reports and the self-referential manifest files are intentionally excluded from the payload manifest.",
    }
    write_json(ROOT / "validation/final_validation_report.json", report)
    lines = ["# Final validation report", "", f"**Status: {status}**", "", "| Gate | Result | Evidence |", "|---|---|---|"]
    for gate in gates:
        lines.append(f"| {gate['gate']} | {'PASS' if gate['pass'] else 'FAIL'} | {str(gate['evidence']).replace('|', '/')} |")
    (ROOT / "validation/final_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
