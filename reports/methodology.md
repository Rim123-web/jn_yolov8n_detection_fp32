# Methodology

This benchmark evaluates the **YOLOv8n six-class object detector** from `/home/jetson/Downloads/jetson_folder/oct_albument2_v6.pt` (SHA-256 `7c09b3b9866f633c60005b7446f29e9612fae00a8789ab9fe983b32b2f08828d`). The source was freshly exported to FP32 ONNX opset 12 and parsed/built with TensorRT 8.0.1.6 from /home/jetson/.local/lib/python3.8/site-packages.

The engine was built in explicit-batch strict FP32 mode: FP16 OFF, INT8 OFF, TF32 OFF, DLA OFF, no calibration, static input `1x3x640x960`, batch 1, 1024 MiB workspace. The six classes are: leccino, varietafrancese, cimadimelfimisto, ulivigrandi, montalbanosecolari, irriguo.

Pillow decodes each image, applies EXIF transpose, converts to RGB, and performs centered bilinear letterbox resizing to 640x960 with padding value 114. Pixels are divided by 255 only, converted HWC-to-NCHW, and made contiguous FP32. No ImageNet mean/std normalization is used.

TensorRT uses one CUDA stream and reusable buffers. CUDA events independently time H2D, `execute_async_v2`, and D2H. Every event end is explicitly synchronized and the stream is synchronized before returning. External CPU custom NumPy class-aware NMS is timed independently. The official deployment confidence is 0.25, NMS IoU is 0.70, and metric matching IoU is 0.50. Low-confidence predictions down to 0.001 are derived offline from raw outputs for AP calculation without additional inference.

Exactly 10 real images, the first ten in deterministic sorted test order, were used for full-pipeline warm-up. Their 10 engine executions and timings are preserved separately and excluded from latency, FPS, telemetry, resources, energy, and metrics. The 192 official test images were then executed exactly once each, with no shuffle, retry, discard, hidden execution, drawing, plotting, or metric calculation inside the timed loop.

jtop/jetson-stats Power TOT was sampled at a requested 0.5-second interval. A 30-sample health gate and a fresh 60-second idle baseline were required. Active telemetry began only after warm-up 10. Gross and idle-corrected energy were calculated by trapezoidal integration of timestamped power samples, with benchmark endpoints extended using the nearest valid sample.

Detection AP uses a 101-point interpolated precision envelope over IoU 0.50:0.05:0.95 and unweighted macro averaging across six classes. YOLO labels contain no ignored/difficult flags. All detailed stage timings, raw outputs, predictions, telemetry, source code, hashes, and execution counts are preserved.
