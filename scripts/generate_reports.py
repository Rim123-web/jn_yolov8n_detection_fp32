#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path

from common import CLASS_NAMES, ROOT, append_timeline, file_metadata, read_json


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    append_timeline("report_generation", "START", "publication reports")
    environment = read_json(ROOT / "environment/system_information.json", {})
    software = read_json(ROOT / "environment/software_versions.json", {})
    model = read_json(ROOT / "model/source_model_metadata.json", {})
    onnx = read_json(ROOT / "onnx/onnx_metadata.json", {})
    engine = read_json(ROOT / "engine/engine_metadata.json", {})
    build = read_json(ROOT / "engine/engine_build_duration.json", {})
    correctness = read_json(ROOT / "validation/fp32_correctness_summary.json", {})
    warmup = read_json(ROOT / "raw/warmup_execution_provenance.json", {})
    execution = read_json(ROOT / "raw/execution_count_provenance.json", {})
    performance = read_json(ROOT / "summary/performance_summary.json", {})
    metrics = read_json(ROOT / "metrics/detection_metrics.json", {})
    energy = read_json(ROOT / "energy/energy_summary.json", {})
    resources = read_json(ROOT / "resources/resource_summary.json", {})
    temperature = read_json(ROOT / "temperature/temperature_summary.json", {})
    throttling = read_json(ROOT / "throttling/throttling_summary.json", {})
    bottleneck = read_json(ROOT / "analysis/bottleneck_summary.json", {})
    idle = read_json(ROOT / "monitoring/idle_summary.json", {})
    timing_rows = {row["stage"]: row for row in __import__("csv").DictReader((ROOT / "summary/timing_stage_summary.csv").open(encoding="utf-8"))}

    methodology = f"""# Methodology

This benchmark evaluates the **YOLOv8n six-class object detector** from `{model.get('path')}` (SHA-256 `{model.get('sha256')}`). The source was freshly exported to FP32 ONNX opset {onnx.get('opsets', [{}])[0].get('version', 'NA')} and parsed/built with TensorRT {software.get('tensorrt')}.

The engine was built in explicit-batch strict FP32 mode: FP16 OFF, INT8 OFF, TF32 OFF, DLA OFF, no calibration, static input `1x3x640x960`, batch 1, 1024 MiB workspace. The six classes are: {', '.join(CLASS_NAMES)}.

Pillow decodes each image, applies EXIF transpose, converts to RGB, and performs centered bilinear letterbox resizing to 640x960 with padding value 114. Pixels are divided by 255 only, converted HWC-to-NCHW, and made contiguous FP32. No ImageNet mean/std normalization is used.

TensorRT uses one CUDA stream and reusable buffers. CUDA events independently time H2D, `execute_async_v2`, and D2H. Every event end is explicitly synchronized and the stream is synchronized before returning. External CPU custom NumPy class-aware NMS is timed independently. The official deployment confidence is 0.25, NMS IoU is 0.70, and metric matching IoU is 0.50. Low-confidence predictions down to 0.001 are derived offline from raw outputs for AP calculation without additional inference.

Exactly 10 real images, the first ten in deterministic sorted test order, were used for full-pipeline warm-up. Their 10 engine executions and timings are preserved separately and excluded from latency, FPS, telemetry, resources, energy, and metrics. The 192 official test images were then executed exactly once each, with no shuffle, retry, discard, hidden execution, drawing, plotting, or metric calculation inside the timed loop.

jtop/jetson-stats Power TOT was sampled at a requested 0.5-second interval. A 30-sample health gate and a fresh 60-second idle baseline were required. Active telemetry began only after warm-up 10. Gross and idle-corrected energy were calculated by trapezoidal integration of timestamped power samples, with benchmark endpoints extended using the nearest valid sample.

Detection AP uses a 101-point interpolated precision envelope over IoU 0.50:0.05:0.95 and unweighted macro averaging across six classes. YOLO labels contain no ignored/difficult flags. All detailed stage timings, raw outputs, predictions, telemetry, source code, hashes, and execution counts are preserved.
"""
    (ROOT / "reports/methodology.md").write_text(methodology, encoding="utf-8")

    results = f"""# Results

## Detection metrics

| Metric | Value |
|---|---:|
| mAP@0.50 | {fmt(metrics.get('mAP50'))} |
| mAP@0.50:0.95 | {fmt(metrics.get('mAP50_95'))} |
| Precision at confidence 0.25 | {fmt(metrics.get('precision_conf025'))} |
| Recall at confidence 0.25 | {fmt(metrics.get('recall_conf025'))} |
| F1 at confidence 0.25 | {fmt(metrics.get('f1_conf025'))} |
| Best F1 | {fmt(metrics.get('best_f1'))} at confidence {fmt(metrics.get('best_f1_confidence_threshold'), 3)} |

## Performance definitions

| Definition | Latency / throughput |
|---|---:|
| Pure TensorRT engine mean | {fmt(performance.get('pure_tensorrt_engine_latency_mean_ms'), 3)} ms |
| Engine FPS | {fmt(performance.get('engine_fps'), 3)} images/s |
| Transfer-inclusive mean | {fmt(performance.get('transfer_inclusive_latency_mean_ms'), 3)} ms |
| Transfer-inclusive FPS | {fmt(performance.get('transfer_inclusive_fps'), 3)} images/s |
| Mean pipeline latency | {fmt(performance.get('mean_pipeline_latency_ms'), 3)} ms |
| Latency-derived pipeline FPS | {fmt(performance.get('latency_derived_pipeline_fps'), 3)} images/s |
| Wall-clock pipeline FPS | {fmt(performance.get('wall_clock_pipeline_fps'), 3)} images/s |

## Energy and resources

| Metric | Value |
|---|---:|
| Idle mean Power TOT | {fmt(energy.get('idle_mean_power_w'), 3)} W |
| Active time-weighted mean Power TOT | {fmt(energy.get('active_mean_power_w_time_weighted'), 3)} W |
| Gross energy | {fmt(energy.get('gross_energy_j'), 3)} J |
| Gross energy/image | {fmt(energy.get('gross_energy_per_image_j'), 3)} J |
| Idle-corrected energy | {fmt(energy.get('idle_corrected_energy_j'), 3)} J |
| Idle-corrected energy/image | {fmt(energy.get('idle_corrected_energy_per_image_j'), 3)} J |
| CPU mean/max | {fmt(resources.get('cpu_mean_percent'), 2)} / {fmt(resources.get('cpu_max_percent'), 2)} % |
| GPU mean/max | {fmt(resources.get('gpu_mean_percent'), 2)} / {fmt(resources.get('gpu_max_percent'), 2)} % |
| RAM mean/max | {fmt(resources.get('ram_mean_mib'), 1)} / {fmt(resources.get('ram_max_mib'), 1)} MiB |
| Process RSS mean/max | {fmt(resources.get('process_rss_mean_mib'), 1)} / {fmt(resources.get('process_rss_max_mib'), 1)} MiB |
| Temperature mean/max | {fmt(temperature.get('mean_c'), 2)} / {fmt(temperature.get('max_c'), 2)} C |
| Throttling events | {throttling.get('event_count', 'NA')} |

The primary bottleneck is `{bottleneck.get('largest_mean_latency_stage')}`. Approximately {fmt(bottleneck.get('percentage_outside_tensorrt'), 2)}% of end-to-end latency occurs outside pure TensorRT engine execution.
"""
    (ROOT / "reports/results.md").write_text(results, encoding="utf-8")

    limitations = """# Limitations

- Jetson Nano hardware, TensorRT 8.0.1, CUDA 10.2, and dynamic clocks limit direct generalization to newer devices.
- The official full pipeline uses Pillow and CPU NumPy NMS; alternative decoders or GPU NMS would change application latency.
- Host memory is pageable rather than pinned; this is documented and H2D/D2H remain independently measured.
- jtop updates may be coarser than the requested 0.5-second polling interval; energy integration uses actual timestamps and reports coverage.
- Power TOT is board-input power, not GPU-only power. Idle correction is reported separately from gross energy.
- mAP is computed by a preserved custom YOLO-label evaluator rather than invoking a hidden second Ultralytics validation benchmark.
- The dataset contains 192 images and one empty-label image; conclusions apply to this fixed test set and class distribution.
"""
    (ROOT / "reports/limitations.md").write_text(limitations, encoding="utf-8")

    reproducibility = f"""# Reproducibility

The package contains the source model copy, fresh ONNX, fresh strict-FP32 TensorRT engine, deterministic 192-image manifest/order, annotation validation, canonical preprocessing/postprocessing/runtime code, exactly 10 warm-up rows, 192 official timing rows, raw outputs and predictions, raw idle/active telemetry, metrics, 33 figures in PNG/SVG/PDF with source CSVs, reports, execution commands, environment versions, and SHA-256 manifests.

Correctness executions ({correctness.get('tensorrt_executions')}) are separate from warm-up ({warmup.get('engine_executions')}) and official measured executions ({execution.get('measured_executions')}). Hidden executions: {execution.get('hidden_executions')}. The original source model is never overwritten.
"""
    (ROOT / "reports/reproducibility.md").write_text(reproducibility, encoding="utf-8")

    article = f"""# Article-ready YOLOv8n TensorRT FP32 results

At static batch-one input `1x3x640x960`, the fresh strict-FP32 TensorRT engine achieved mAP@0.50 **{fmt(metrics.get('mAP50'))}**, mAP@0.50:0.95 **{fmt(metrics.get('mAP50_95'))}**, and F1@0.25 **{fmt(metrics.get('f1_conf025'))}**. Mean pure-engine latency was **{fmt(performance.get('pure_tensorrt_engine_latency_mean_ms'), 3)} ms** ({fmt(performance.get('engine_fps'), 3)} engine FPS), while mean complete pipeline latency was **{fmt(performance.get('mean_pipeline_latency_ms'), 3)} ms** and measured wall-clock pipeline throughput was **{fmt(performance.get('wall_clock_pipeline_fps'), 3)} images/s**. Gross board energy was **{fmt(energy.get('gross_energy_per_image_j'), 3)} J/image**, and idle-corrected energy was **{fmt(energy.get('idle_corrected_energy_per_image_j'), 3)} J/image**. Ten real-image full-pipeline warm-ups were excluded from every official result.
"""
    (ROOT / "reports/article_ready_results.md").write_text(article, encoding="utf-8")
    defense = f"""# Defense-ready summary

- Model: YOLOv8n, six-class detection, strict TensorRT FP32.
- Shape: `1x3x640x960`, batch 1.
- Protocol: 20-image correctness gate, exactly 10 excluded real-image warm-ups, 192 images measured once.
- Accuracy: mAP50 {fmt(metrics.get('mAP50'))}; mAP50-95 {fmt(metrics.get('mAP50_95'))}; F1@0.25 {fmt(metrics.get('f1_conf025'))}.
- Speed: engine {fmt(performance.get('engine_fps'), 3)} FPS; wall-clock pipeline {fmt(performance.get('wall_clock_pipeline_fps'), 3)} FPS.
- Energy: gross {fmt(energy.get('gross_energy_per_image_j'), 3)} J/image; idle-corrected {fmt(energy.get('idle_corrected_energy_per_image_j'), 3)} J/image.
- Main bottleneck: {bottleneck.get('largest_mean_latency_stage')}; {fmt(bottleneck.get('percentage_outside_tensorrt'), 2)}% outside engine.
- No failed, retried, discarded, or hidden official executions.
"""
    (ROOT / "reports/defense_ready_summary.md").write_text(defense, encoding="utf-8")
    (ROOT / "reports/failure_and_recovery_log.md").write_text(
        "# Failure and recovery log\n\n"
        "- Package preparation completed successfully. A trailing shell-only `grep` for an unavailable `/proc/vmstat` field returned nonzero after preparation had passed; the empty evidence file and kernel journal evidence were preserved.\n"
        "- The first TensorRT parser launcher failed before parsing because the clean Python environment filtered the TensorRT user-site path. Its traceback was preserved as `logs/tensorrt_parser_import_failure.log`. The package-local scripts were corrected to insert only the known TensorRT Python directory at runtime, and the actual parser gate was then executed once.\n"
        "- The first engine-builder launcher terminated with SIGILL (exit 132) while importing its module, before entering `main()` or invoking any TensorRT builder API. An import-only probe reproduced the issue and identified the Jetson-specific NumPy/TensorRT initialization order. The empty engine directory and original one-line console log were preserved; the package-local builder was changed to initialize NumPy before TensorRT, then verified with an import-only probe before the first actual build call.\n"
        "- The first actual builder call completed and serialized the fresh strict-FP32 engine, then the same process failed during post-build metadata validation because TensorRT 8.0.1 `trt.nptype()` references the removed NumPy 1.24 alias `np.bool`. The engine was not rebuilt. The package runtime replaced only that deprecated dtype helper with an explicit TensorRT-to-NumPy mapping, and a separate clean no-inference process completed deserialization, binding, shape, and dtype validation.\n"
        "- The 20-image correctness gate completed with 20/20 console passes and saved all backend predictions, then failed while serializing a NumPy boolean into its JSON summary. Those 20 executions were not repeated. The summary was reconstructed offline from the saved prediction CSV and original 20 pass lines; raw-difference values that were not serialized are explicitly marked unavailable rather than guessed. The shared JSON writer was made NumPy-scalar compatible for subsequent phases.\n"
        "- A pre-acceptance second-pass audit found that preprocessing stage durations from `perf_counter_ns()` had been multiplied by 1000 while labeled milliseconds. Engine, transfer, postprocessing, end-to-end, wall-clock, telemetry, energy, predictions, and metrics were unaffected. Original timing CSVs were preserved under `raw/pre_timing_unit_correction/`; preprocessing fields in warm-up and official CSVs were corrected offline by the exact factor 1e-9, dependent analyses/figures/reports were regenerated, and the final validator gained a per-row unit/decomposition gate. No inference was rerun.\n"
        "- A third-pass report audit found that the bottleneck selector compared atomic stages with aggregate totals and therefore selected transfer-inclusive inference. The selector was corrected offline to compare independently timed atomic stages only; the TensorRT engine is the resulting atomic bottleneck. No measured value changed and no inference was rerun.\n"
        "- The ONNX export SSH control channel timed out after launch, but the single remote exporter continued and completed successfully; no duplicate export was started.\n"
        "- No official benchmark execution was retried, discarded, or hidden. ONNX and ONNXRuntime were supplied by the existing `jetson_benchmark_env` through a recorded PYTHONPATH; no package was installed or upgraded. PyCUDA remained unavailable and was intentionally not used; CUDA Runtime ctypes provided allocation, copies, streams, events, and synchronization.\n",
        encoding="utf-8",
    )
    full = f"""# Full YOLOv8n TensorRT FP32 benchmark report

{methodology}

{results}

{limitations}

{reproducibility}
"""
    (ROOT / "reports/full_benchmark_report.md").write_text(full, encoding="utf-8")
    (ROOT / "reports/full_benchmark_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>YOLOv8n TensorRT FP32 Benchmark</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.5}pre{white-space:pre-wrap}</style></head>"
        f"<body><pre>{html.escape(full)}</pre></body></html>\n",
        encoding="utf-8",
    )
    append_timeline("report_generation", "PASS", "9 required reports created")


if __name__ == "__main__":
    main()
