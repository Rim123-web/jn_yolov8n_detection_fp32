# Article-ready YOLOv8n TensorRT FP32 results

At static batch-one input `1x3x640x960`, the fresh strict-FP32 TensorRT engine achieved mAP@0.50 **0.977236**, mAP@0.50:0.95 **0.830485**, and F1@0.25 **0.918420**. Mean pure-engine latency was **91.636 ms** (10.913 engine FPS), while mean complete pipeline latency was **148.078 ms** and measured wall-clock pipeline throughput was **6.622 images/s**. Gross board energy was **1.082 J/image**, and idle-corrected energy was **0.686 J/image**. Ten real-image full-pipeline warm-ups were excluded from every official result.
