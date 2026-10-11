#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

from common import CLASS_NAMES, HEIGHT, ROOT, SEED, WIDTH, append_timeline, collect_images, file_metadata, write_csv, write_json
from postprocessing import postprocess_yolov8
from preprocessing import preprocess_image


def raw_torch_output(model, array):
    tensor = torch.from_numpy(array).to("cuda:0", dtype=torch.float32)
    torch.cuda.synchronize()
    with torch.inference_mode():
        output = model(tensor)
    torch.cuda.synchronize()
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().numpy()


def compare_detections(a, b):
    a = sorted(a, key=lambda x: (-x["confidence"], x["class_id"]))
    b = sorted(b, key=lambda x: (-x["confidence"], x["class_id"]))
    paired = min(len(a), len(b))
    class_agree = sum(a[i]["class_id"] == b[i]["class_id"] for i in range(paired))
    confidence_diff = max((abs(a[i]["confidence"] - b[i]["confidence"]) for i in range(paired)), default=0.0)
    box_diff = max((max(abs(x - y) for x, y in zip(a[i]["box"], b[i]["box"])) for i in range(paired)), default=0.0)
    return class_agree, paired, confidence_diff, box_diff


def main():
    append_timeline("onnx_validation", "START", "checker + 20 real-image equivalence")
    path = ROOT / "onnx/oct_albument2_v6_inference_only.onnx"
    model_proto = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model_proto)
    graph_inputs = []
    for value in model_proto.graph.input:
        graph_inputs.append({
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim],
        })
    graph_outputs = []
    for value in model_proto.graph.output:
        graph_outputs.append({
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim],
        })
    operators = Counter(node.op_type for node in model_proto.graph.node)
    write_csv(ROOT / "onnx/onnx_graph_operator_inventory.csv", [{"operator": k, "count": v} for k, v in sorted(operators.items())])
    nms_ops = {k: v for k, v in operators.items() if "nms" in k.lower() or "nonmax" in k.lower()}
    metadata = {
        **file_metadata(path), "onnx_version": onnx.__version__, "ir_version": model_proto.ir_version,
        "opsets": [{"domain": x.domain, "version": x.version} for x in model_proto.opset_import],
        "inputs": graph_inputs, "outputs": graph_outputs, "node_count": len(model_proto.graph.node),
        "initializer_count": len(model_proto.graph.initializer),
        "dynamic_dimensions": any(isinstance(x, str) and x for item in graph_inputs + graph_outputs for x in item["shape"]),
    }
    write_json(ROOT / "onnx/onnx_metadata.json", metadata)
    (ROOT / "onnx/onnx_sha256.txt").write_text(f"{metadata['sha256']}  {path.name}\n", encoding="utf-8")
    write_json(ROOT / "onnx/nms_graph_inspection.json", {
        "nms_nodes": nms_ops, "embedded_nms": bool(nms_ops), "status": "PASS" if not nms_ops else "FAIL",
        "raw_output_expected": True, "output_shapes": graph_outputs,
    })
    expected_shape = [1, 3, HEIGHT, WIDTH]
    checker_pass = len(graph_inputs) == 1 and graph_inputs[0]["shape"] == expected_shape and graph_inputs[0]["dtype"] == "FLOAT" and not nms_ops and not metadata["dynamic_dimensions"]
    write_json(ROOT / "validation/onnx_checker_report.json", {
        "status": "PASS" if checker_pass else "FAIL", "onnx_checker": "PASS",
        "input_shape": graph_inputs, "outputs": graph_outputs, "batch": 1,
        "input_dtype_fp32": graph_inputs[0]["dtype"] == "FLOAT", "raw_output": True,
        "embedded_nms": bool(nms_ops), "dynamic_dimensions": metadata["dynamic_dimensions"],
    })
    if not checker_pass:
        raise RuntimeError("ONNX checker/metadata gate failed")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch_model = YOLO(str(ROOT / "model/oct_albument2_v6_source.pt")).model.to("cuda:0").float().eval()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows = []
    for index, image_path in enumerate(collect_images()[:20], 1):
        array, meta, _ = preprocess_image(image_path)
        source = raw_torch_output(torch_model, array)
        onnx_output = session.run([output_name], {input_name: array})[0]
        difference = np.abs(source.astype(np.float32) - onnx_output.astype(np.float32))
        source_dets, _ = postprocess_yolov8(source, meta)
        onnx_dets, _ = postprocess_yolov8(onnx_output, meta)
        class_agree, paired, conf_diff, box_diff = compare_detections(source_dets, onnx_dets)
        passed = source.shape == onnx_output.shape and float(difference.mean()) <= 1e-4 and float(difference.max()) <= 1e-2 and abs(len(source_dets) - len(onnx_dets)) <= 1 and class_agree == paired
        rows.append({
            "sample_index": index, "image_path": str(image_path),
            "source_output_shape": str(tuple(source.shape)), "onnx_output_shape": str(tuple(onnx_output.shape)),
            "maximum_absolute_difference": float(difference.max()), "mean_absolute_difference": float(difference.mean()),
            "source_detection_count": len(source_dets), "onnx_detection_count": len(onnx_dets),
            "paired_detections": paired, "class_agreement_count": class_agree,
            "maximum_confidence_difference": conf_diff, "maximum_box_coordinate_difference": box_diff,
            "pass": passed,
        })
        print(f"ONNX equivalence {index}/20 pass={passed}", flush=True)
    write_csv(ROOT / "validation/onnx_equivalence.csv", rows)
    all_pass = len(rows) == 20 and all(row["pass"] for row in rows)
    write_json(ROOT / "validation/onnx_equivalence_summary.json", {
        "status": "PASS" if all_pass else "FAIL", "samples": len(rows), "passed": sum(bool(x["pass"]) for x in rows),
        "source_pt_executions": 20, "onnxruntime_executions": 20,
        "onnxruntime_version": ort.__version__, "providers": session.get_providers(),
        "max_absolute_difference": max(x["maximum_absolute_difference"] for x in rows),
        "mean_absolute_difference": float(np.mean([x["mean_absolute_difference"] for x in rows])),
        "tolerances": {"max_abs": 1e-2, "mean_abs": 1e-4, "detection_count_delta": 1},
    })
    append_timeline("onnx_validation", "PASS" if all_pass else "FAIL", "20 representative real images")
    if not all_pass:
        raise RuntimeError("ONNX equivalence gate failed")


if __name__ == "__main__":
    main()
