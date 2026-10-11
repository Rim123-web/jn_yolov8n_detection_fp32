# Limitations

- Jetson Nano hardware, TensorRT 8.0.1, CUDA 10.2, and dynamic clocks limit direct generalization to newer devices.
- The official full pipeline uses Pillow and CPU NumPy NMS; alternative decoders or GPU NMS would change application latency.
- Host memory is pageable rather than pinned; this is documented and H2D/D2H remain independently measured.
- jtop updates may be coarser than the requested 0.5-second polling interval; energy integration uses actual timestamps and reports coverage.
- Power TOT is board-input power, not GPU-only power. Idle correction is reported separately from gross energy.
- mAP is computed by a preserved custom YOLO-label evaluator rather than invoking a hidden second Ultralytics validation benchmark.
- The dataset contains 192 images and one empty-label image; conclusions apply to this fixed test set and class distribution.
