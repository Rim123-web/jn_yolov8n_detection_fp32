# Final validation report

**Status: PASS**

| Gate | Result | Evidence |
|---|---|---|
| source_pt_exists | PASS | /home/jetson/Downloads/jetson_folder/yolov8n_detection_fp32_benchmark/yolov8n_detection_tensorrt_fp32_640x960_batch1_warmup10_20260720_165642/model/oct_albument2_v6_source.pt |
| source_hash_matches_original | PASS | 7c09b3b9866f633c60005b7446f29e9612fae00a8789ab9fe983b32b2f08828d |
| onnx_exists_and_hash_matches | PASS | 6a58292b377d7bdfe208fa23adff170e864d9d8695dba0c7ab4d6e426ff15524 |
| engine_exists_and_hash_matches | PASS | 8c050c451554bd664b56b7cde3ed5de7ce02358e56a5f90806f8ee36a6e3d1ab |
| strict_fp32 | PASS |  |
| input_shape | PASS |  |
| batch_one | PASS |  |
| class_mapping | PASS |  |
| dataset_manifest | PASS | rows=192 |
| warmup_rows_exactly_10 | PASS |  |
| warmup_paths_unique | PASS |  |
| warmup_engine_executions_10 | PASS |  |
| warmups_excluded | PASS |  |
| measured_rows_exact | PASS |  |
| prediction_image_rows_exact | PASS |  |
| measured_paths_unique | PASS |  |
| failed_images_zero | PASS |  |
| discarded_rows_zero | PASS |  |
| hidden_executions_zero | PASS |  |
| official_execution_count | PASS |  |
| all_timings_finite_nonnegative | PASS |  |
| independent_nms_timing | PASS |  |
| independent_engine_timing | PASS |  |
| end_to_end_timing | PASS |  |
| timing_units_and_decomposition | PASS | preprocess <2000 ms and /end-to-end - preprocess - transfer-inclusive - postprocess/ <10 ms for every row |
| onnx_checker | PASS |  |
| onnx_equivalence | PASS |  |
| tensorrt_parser | PASS |  |
| correctness | PASS |  |
| jtop_health | PASS |  |
| idle_baseline | PASS |  |
| energy_integration | PASS |  |
| metrics | PASS |  |
| telemetry_coverage | PASS |  |
| figures_all_formats | PASS | {'png': 33, 'svg': 33, 'pdf': 33} |
| figure_source_data | PASS |  |
| required_scripts | PASS |  |
| required_reports | PASS |  |
| timing_unit_repair_evidence | PASS |  |
| atomic_bottleneck_definition | PASS |  |
| hash_manifest | PASS | entries=303 failures=[] |
