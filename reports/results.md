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
