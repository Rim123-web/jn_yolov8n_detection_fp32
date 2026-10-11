# Bottleneck analysis

The largest measured stage is `tensorrt_engine_ms` at 91.636 ms. TensorRT engine execution accounts for 61.88% of mean end-to-end latency; 38.12% is outside the TensorRT engine. NMS alone contributes 1.94%.
