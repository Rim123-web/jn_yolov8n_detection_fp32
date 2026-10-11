#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, ROOT, append_timeline, write_csv, write_json


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(name, fig, rows):
    write_csv(ROOT / f"figures/data/{name}.csv", rows)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"figures/{extension}/{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar_figure(name, title, xlabel, ylabel, labels, values, rows=None, color="#2563eb"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    save_figure(name, fig, rows or [{xlabel: x, ylabel: y} for x, y in zip(labels, values)])


def histogram(name, title, field_label, values, bins=25):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#0f766e", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(field_label)
    ax.set_ylabel("Image count")
    ax.grid(axis="y", alpha=0.25)
    save_figure(name, fig, [{"measured_index": i + 1, field_label: v} for i, v in enumerate(values)])


def line_figure(name, title, xlabel, ylabel, series, rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, x, y in series:
        ax.plot(x, y, label=label, linewidth=1.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend()
    save_figure(name, fig, rows)


def matrix_figure(name, title, matrix_rows, normalized=False):
    labels = CLASS_NAMES + ["background"]
    matrix = np.asarray([[float(row[label]) for label in labels] for row in matrix_rows])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, label="Normalized proportion" if normalized else "Count")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    save_figure(name, fig, matrix_rows)


def main():
    append_timeline("figure_generation", "START", "33 figures x PNG/SVG/PDF")
    timings = read_csv(ROOT / "raw/per_image_timings.csv")
    stage_summary = read_csv(ROOT / "summary/timing_stage_summary.csv")
    performance = json.loads((ROOT / "summary/performance_summary.json").read_text())
    metrics = json.loads((ROOT / "metrics/detection_metrics.json").read_text())
    energy = json.loads((ROOT / "energy/energy_summary.json").read_text())
    active = read_csv(ROOT / "monitoring/active_samples.csv")
    curve = read_csv(ROOT / "metrics/f1_confidence_curve.csv")
    per_class = read_csv(ROOT / "metrics/per_class_metrics.csv")
    ap_iou = read_csv(ROOT / "metrics/ap_by_iou.csv")
    errors = read_csv(ROOT / "metrics/error_analysis.csv")
    confusion = read_csv(ROOT / "metrics/confusion_matrix.csv")
    confusion_norm = read_csv(ROOT / "metrics/confusion_matrix_normalized.csv")
    warmup = read_csv(ROOT / "raw/warmup_timings.csv")
    manifest = read_csv(ROOT / "dataset/test_image_manifest.csv")

    detailed = [r for r in stage_summary if r["stage"] not in {"preprocessing_total_ms", "transfer_inclusive_inference_ms", "postprocessing_total_ms", "end_to_end_ms"}]
    bar_figure("01_per_stage_mean_latency", "YOLOv8n TensorRT FP32 per-stage mean latency", "Pipeline stage", "Mean latency (ms)", [r["stage"].replace("_ms", "") for r in detailed], [float(r["mean_ms"]) for r in detailed], detailed)
    bar_figure("02_per_stage_e2e_percentage", "Per-stage share of end-to-end latency", "Pipeline stage", "Share of E2E (%)", [r["stage"].replace("_ms", "") for r in detailed], [float(r["percentage_of_end_to_end"]) for r in detailed], detailed, "#7c3aed")
    histogram("03_latency_distribution", "Full-pipeline latency distribution", "End-to-end latency (ms)", [float(r["end_to_end_ms"]) for r in timings])
    histogram("04_engine_latency_distribution", "TensorRT FP32 engine latency distribution", "TensorRT engine latency (ms)", [float(r["tensorrt_engine_ms"]) for r in timings])
    histogram("05_e2e_latency_distribution", "End-to-end latency distribution", "End-to-end latency (ms)", [float(r["end_to_end_ms"]) for r in timings], 30)
    runtime_labels = ["H2D", "TensorRT engine", "D2H"]
    runtime_values = [np.mean([float(r[f]) for r in timings]) for f in ("h2d_ms", "tensorrt_engine_ms", "d2h_ms")]
    bar_figure("06_h2d_engine_d2h", "CUDA transfer and engine latency", "Runtime component", "Mean latency (ms)", runtime_labels, runtime_values)
    pre_fields = ["file_access_decode_ms", "exif_orientation_ms", "color_conversion_ms", "resize_calculation_ms", "image_resize_ms", "padding_letterbox_ms", "pixel_scaling_ms", "layout_conversion_ms", "fp32_contiguous_ms"]
    bar_figure("07_preprocessing_decomposition", "Preprocessing decomposition", "Preprocessing stage", "Mean latency (ms)", [x.replace("_ms", "") for x in pre_fields], [np.mean([float(r[x]) for r in timings]) for x in pre_fields])
    post_fields = ["raw_output_decode_ms", "box_conversion_ms", "confidence_calculation_ms", "confidence_filtering_ms", "class_selection_ms", "nms_preparation_ms", "nms_ms", "coordinate_rescaling_ms", "box_clipping_ms", "result_formatting_ms"]
    bar_figure("08_postprocessing_decomposition", "Postprocessing decomposition", "Postprocessing stage", "Mean latency (ms)", [x.replace("_ms", "") for x in post_fields], [np.mean([float(r[x]) for r in timings]) for x in post_fields], color="#dc2626")
    histogram("09_nms_latency_distribution", "External class-aware NumPy NMS latency", "NMS latency (ms)", [float(r["nms_ms"]) for r in timings])
    fps_rows = [{"definition": k, "fps": performance[k]} for k in ("engine_fps", "transfer_inclusive_fps", "latency_derived_pipeline_fps", "wall_clock_pipeline_fps")]
    bar_figure("10_fps_definitions", "FP32 throughput definitions", "Definition", "Images/s", [r["definition"] for r in fps_rows], [r["fps"] for r in fps_rows], fps_rows)
    elapsed = [float(r["monotonic_s"]) - float(active[0]["monotonic_s"]) for r in active]
    line_figure("11_power_over_time", "Active Power TOT during official measured pass", "Time from first active sample (s)", "Power TOT (W)", [("Power TOT", elapsed, [float(r["power_tot_w"]) for r in active])], active)
    line_figure("12_cpu_gpu_utilization", "CPU and GPU utilization during official pass", "Time (s)", "Utilization (%)", [("CPU", elapsed, [float(r["cpu_utilization_percent"]) for r in active]), ("GPU", elapsed, [float(r["gpu_utilization_percent"]) for r in active])], active)
    line_figure("13_ram_process_rss", "System RAM and benchmark-process RSS", "Time (s)", "Memory (MiB)", [("System RAM used", elapsed, [float(r["ram_used_mib"]) for r in active]), ("Process RSS", elapsed, [float(r["process_rss_mib"]) for r in active])], active)
    line_figure("14_temperature_over_time", "Maximum reported Jetson temperature", "Time (s)", "Temperature (C)", [("Temperature", elapsed, [float(r["temperature_c"]) for r in active])], active)
    energy_rows = [{"metric": "Gross", "energy_j": energy["gross_energy_j"]}, {"metric": "Idle-corrected", "energy_j": energy["idle_corrected_energy_j"]}]
    bar_figure("15_energy_summary", "Energy over the official 192-image pass", "Energy definition", "Energy (J)", [r["metric"] for r in energy_rows], [r["energy_j"] for r in energy_rows], energy_rows, "#ea580c")
    line_figure("16_precision_recall_curve", "Precision-recall curve from saved FP32 predictions", "Recall", "Precision", [("TensorRT FP32", [float(r["recall"]) for r in curve], [float(r["precision"]) for r in curve])], curve)
    line_figure("17_f1_confidence_curve", "F1 versus confidence threshold", "Confidence threshold", "F1", [("TensorRT FP32", [float(r["confidence_threshold"]) for r in curve], [float(r["f1"]) for r in curve])], curve)
    bar_figure("18_ap_by_class", "Average precision by class", "Class", "AP@0.50:0.95", [r["class_name"] for r in per_class], [float(r["ap50_95"]) for r in per_class], per_class)
    grouped = defaultdict(list)
    for row in ap_iou:
        grouped[float(row["iou_threshold"])].append(float(row["ap"]))
    iou_rows = [{"iou_threshold": key, "mean_ap": np.mean(value)} for key, value in sorted(grouped.items())]
    line_figure("19_ap_by_iou", "Mean AP by IoU threshold", "IoU threshold", "Mean AP", [("Mean AP", [r["iou_threshold"] for r in iou_rows], [r["mean_ap"] for r in iou_rows])], iou_rows)
    histogram("20_detection_count_distribution", "Detection count per image at confidence 0.25", "Detection count", [float(r["detection_count"]) for r in timings], bins=15)
    error_rows = [{"category": "Images with FP", "count": sum(r["has_false_positive"].lower() == "true" for r in errors)}, {"category": "Images with FN", "count": sum(r["has_false_negative"].lower() == "true" for r in errors)}]
    bar_figure("21_error_analysis", "Detection error analysis", "Error category", "Image count", [r["category"] for r in error_rows], [r["count"] for r in error_rows], error_rows, "#b91c1c")
    matrix_figure("22_confusion_matrix", "Detection confusion matrix at confidence 0.25", confusion)
    matrix_figure("23_confusion_matrix_normalized", "Normalized detection confusion matrix", confusion_norm, True)
    line_figure("24_warmup_timing_behavior", "Ten excluded real-image warm-ups", "Warm-up index", "Latency (ms)", [("Engine", [int(r["warmup_index"]) for r in warmup], [float(r["tensorrt_engine_ms"]) for r in warmup]), ("E2E", [int(r["warmup_index"]) for r in warmup], [float(r["end_to_end_ms"]) for r in warmup])], warmup)
    resolution_counts = Counter((int(r["width"]), int(r["height"])) for r in manifest)
    resolution_rows = [{"resolution": f"{w}x{h}", "image_count": count} for (w, h), count in resolution_counts.items()]
    bar_figure("25_original_resolution_distribution", "Original test-image resolution distribution", "Resolution", "Image count", [r["resolution"] for r in resolution_rows], [r["image_count"] for r in resolution_rows], resolution_rows)
    line_figure("26_letterbox_padding_overhead", "Letterbox/padding overhead by measured image", "Measured image index", "Padding latency (ms)", [("Padding", list(range(1, len(timings) + 1)), [float(r["padding_letterbox_ms"]) for r in timings])], timings)
    histogram("27_decode_time_distribution", "Image file-access and decode latency", "Decode latency (ms)", [float(r["file_access_decode_ms"]) for r in timings])
    histogram("28_resize_time_distribution", "PIL bilinear resize latency", "Resize latency (ms)", [float(r["image_resize_ms"]) for r in timings])
    histogram("29_coordinate_rescaling_distribution", "Coordinate-rescaling latency", "Coordinate-rescaling latency (ms)", [float(r["coordinate_rescaling_ms"]) for r in timings])
    box_fields = ["file_access_decode_ms", "image_resize_ms", "h2d_ms", "tensorrt_engine_ms", "d2h_ms", "nms_ms"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([[float(r[field]) for r in timings] for field in box_fields], labels=[x.replace("_ms", "") for x in box_fields], showfliers=True)
    ax.set_title("Pipeline-stage latency boxplot")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=35)
    save_figure("30_stage_boxplot", fig, [{"measured_index": i + 1, **{field: timings[i][field] for field in box_fields}} for i in range(len(timings))])
    cumulative_rows = []
    cumulative = 0.0
    for row in sorted(detailed, key=lambda x: float(x["mean_ms"]), reverse=True):
        cumulative += float(row["percentage_of_end_to_end"])
        cumulative_rows.append({"stage": row["stage"], "mean_ms": float(row["mean_ms"]), "cumulative_percent": cumulative})
    line_figure("31_cumulative_latency_contribution", "Cumulative latency contribution", "Stages ordered by mean contribution", "Cumulative share of E2E (%)", [("Cumulative", list(range(1, len(cumulative_rows) + 1)), [r["cumulative_percent"] for r in cumulative_rows])], cumulative_rows)
    accuracy_rows = [{"metric": "mAP@50", "value": metrics["mAP50"]}, {"metric": "mAP@50-95", "value": metrics["mAP50_95"]}, {"metric": "F1@0.25", "value": metrics["f1_conf025"]}]
    bar_figure("32_accuracy_performance_summary", "TensorRT FP32 detection accuracy", "Metric", "Score", [r["metric"] for r in accuracy_rows], [r["value"] for r in accuracy_rows], accuracy_rows, "#16a34a")
    deployment_rows = [{"metric": "Engine FPS", "value": performance["engine_fps"]}, {"metric": "Pipeline FPS", "value": performance["wall_clock_pipeline_fps"]}, {"metric": "mAP50 x100", "value": metrics["mAP50"] * 100}, {"metric": "Energy J/image x10", "value": energy["gross_energy_per_image_j"] * 10}]
    bar_figure("33_deployment_summary", "YOLOv8n strict-FP32 deployment summary", "Deployment metric", "Scaled value (see labels)", [r["metric"] for r in deployment_rows], [r["value"] for r in deployment_rows], deployment_rows, "#334155")
    names = [f"{i:02d}_" for i in range(1, 34)]
    counts = {extension: len(list((ROOT / f"figures/{extension}").glob(f"*.{extension}"))) for extension in ("png", "svg", "pdf")}
    data_count = len(list((ROOT / "figures/data").glob("*.csv")))
    passed = all(counts[x] == 33 for x in counts) and data_count == 33
    write_json(ROOT / "figures/figure_generation_summary.json", {"status": "PASS" if passed else "FAIL", "counts": counts, "source_csv_count": data_count})
    append_timeline("figure_generation", "PASS" if passed else "FAIL", json.dumps(counts))
    if not passed:
        raise RuntimeError(f"Figure generation count mismatch: {counts}, data={data_count}")


if __name__ == "__main__":
    main()
