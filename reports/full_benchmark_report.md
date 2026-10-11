# Full YOLOv8n TensorRT FP32 benchmark report

# Methodology

This benchmark evaluates the **YOLOv8n six-class object detector** from `/home/jetson/Downloads/jetson_folder/oct_albument2_v6.pt` (SHA-256 `7c09b3b9866f633c60005b7446f29e9612fae00a8789ab9fe983b32b2f08828d`). The source was freshly exported to FP32 ONNX opset 12 and parsed/built with TensorRT 8.0.1.6 from /home/jetson/.local/lib/python3.8/site-packages.

The engine was built in explicit-batch strict FP32 mode: FP16 OFF, INT8 OFF, TF32 OFF, DLA OFF, no calibration, static input `1x3x640x960`, batch 1, 1024 MiB workspace. The six classes are: leccino, varietafrancese, cimadimelfimisto, ulivigrandi, montalbanosecolari, irriguo.

Pillow decodes each image, applies EXIF transpose, converts to RGB, and performs centered bilinear letterbox resizing to 640x960 with padding value 114. Pixels are divided by 255 only, converted HWC-to-NCHW, and made contiguous FP32. No ImageNet mean/std normalization is used.

TensorRT uses one CUDA stream and reusable buffers. CUDA events independently time H2D, `execute_async_v2`, and D2H. Every event end is explicitly synchronized and the stream is synchronized before returning. External CPU custom NumPy class-aware NMS is timed independently. The official deployment confidence is 0.25, NMS IoU is 0.70, and metric matching IoU is 0.50. Low-confidence predictions down to 0.001 are derived offline from raw outputs for AP calculation without additional inference.

Exactly 10 real images, the first ten in deterministic sorted test order, were used for full-pipeline warm-up. Their 10 engine executions and timings are preserved separately and excluded from latency, FPS, telemetry, resources, energy, and metrics. The 192 official test images were then executed exactly once each, with no shuffle, retry, discard, hidden execution, drawing, plotting, or metric calculation inside the timed loop.

jtop/jetson-stats Power TOT was sampled at a requested 0.5-second interval. A 30-sample health gate and a fresh 60-second idle baseline were required. Active telemetry began only after warm-up 10. Gross and idle-corrected energy were calculated by trapezoidal integration of timestamped power samples, with benchmark endpoints extended using the nearest valid sample.

Detection AP uses a 101-point interpolated precision envelope over IoU 0.50:0.05:0.95 and unweighted macro averaging across six classes. YOLO labels contain no ignored/difficult flags. All detailed stage timings, raw outputs, predictions, telemetry, source code, hashes, and execution counts are preserved.


# Results

## Detection metrics

| Metric | Value |
|---|---:|
| mAP@0.50 | 0.977236 |
| mAP@0.50:0.95 | 0.830485 |
| Precision at confidence 0.25 | 0.868904 |
| Recall at confidence 0.25 | 0.973922 |
| F1 at confidence 0.25 | 0.918420 |
| Best F1 | 0.947155 at confidence 0.740 |

## Performance definitions

| Definition | Latency / throughput |
|---|---:|
| Pure TensorRT engine mean | 91.636 ms |
| Engine FPS | 10.913 images/s |
| Transfer-inclusive mean | 96.902 ms |
| Transfer-inclusive FPS | 10.320 images/s |
| Mean pipeline latency | 148.078 ms |
| Latency-derived pipeline FPS | 6.753 images/s |
| Wall-clock pipeline FPS | 6.622 images/s |

## Energy and resources

| Metric | Value |
|---|---:|
| Idle mean Power TOT | 2.618 W |
| Active time-weighted mean Power TOT | 7.162 W |
| Gross energy | 207.668 J |
| Gross energy/image | 1.082 J |
| Idle-corrected energy | 131.753 J |
| Idle-corrected energy/image | 0.686 J |
| CPU mean/max | 37.34 / 59.25 % |
| GPU mean/max | 64.02 / 99.70 % |
| RAM mean/max | 2459.9 / 2557.8 MiB |
| Process RSS mean/max | 1432.8 / 1544.1 MiB |
| Temperature mean/max | 52.70 / 55.00 C |
| Throttling events | 0 |

The primary bottleneck is `tensorrt_engine_ms`. Approximately 38.12% of end-to-end latency occurs outside pure TensorRT engine execution.


# Limitations

- Jetson Nano hardware, TensorRT 8.0.1, CUDA 10.2, and dynamic clocks limit direct generalization to newer devices.
- The official full pipeline uses Pillow and CPU NumPy NMS; alternative decoders or GPU NMS would change application latency.
- Host memory is pageable rather than pinned; this is documented and H2D/D2H remain independently measured.
- jtop updates may be coarser than the requested 0.5-second polling interval; energy integration uses actual timestamps and reports coverage.
- Power TOT is board-input power, not GPU-only power. Idle correction is reported separately from gross energy.
- mAP is computed by a preserved custom YOLO-label evaluator rather than invoking a hidden second Ultralytics validation benchmark.
- The dataset contains 192 images and one empty-label image; conclusions apply to this fixed test set and class distribution.


# Reproducibility

The package contains the source model copy, fresh ONNX, fresh strict-FP32 TensorRT engine, deterministic 192-image manifest/order, annotation validation, canonical preprocessing/postprocessing/runtime code, exactly 10 warm-up rows, 192 official timing rows, raw outputs and predictions, raw idle/active telemetry, metrics, 33 figures in PNG/SVG/PDF with source CSVs, reports, execution commands, environment versions, and SHA-256 manifests.

Correctness executions (20) are separate from warm-up (10) and official measured executions (192). Hidden executions: 0. The original source model is never overwritten.

