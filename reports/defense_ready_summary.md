# Defense-ready summary

- Model: YOLOv8n, six-class detection, strict TensorRT FP32.
- Shape: `1x3x640x960`, batch 1.
- Protocol: 20-image correctness gate, exactly 10 excluded real-image warm-ups, 192 images measured once.
- Accuracy: mAP50 0.977236; mAP50-95 0.830485; F1@0.25 0.918420.
- Speed: engine 10.913 FPS; wall-clock pipeline 6.622 FPS.
- Energy: gross 1.082 J/image; idle-corrected 0.686 J/image.
- Main bottleneck: tensorrt_engine_ms; 38.12% outside engine.
- No failed, retried, discarded, or hidden official executions.
