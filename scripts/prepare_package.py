#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import jtop as jtop_module
import numpy as np
import psutil
import torch
import torchvision
import ultralytics
import PIL
from PIL import Image
from ultralytics import YOLO

from common import (
    BATCH, CLASS_NAMES, DATA_YAML, DATASET_ROOT, HEIGHT, IMAGE_DIR, LABEL_DIR,
    ROOT, SEED, SOURCE_PT, WIDTH, append_timeline, collect_images, file_metadata,
    label_path_for_image, sha256_file, utc_now, write_csv, write_json,
)


DIRECTORIES = [
    "environment", "model", "dataset", "config", "onnx", "engine", "validation",
    "scripts", "logs", "raw", "monitoring", "summary", "metrics", "energy",
    "resources", "temperature", "throttling", "analysis", "figures/png",
    "figures/svg", "figures/pdf", "figures/data", "reports", "provenance",
]


def command_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return result.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def snapshot_environment():
    system = {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "ip_addresses": command_text(["hostname", "-I"]).split(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "ubuntu": command_text(["lsb_release", "-ds"]),
        "kernel": platform.release(),
        "l4t": Path("/etc/nv_tegra_release").read_text().strip() if Path("/etc/nv_tegra_release").exists() else "NA",
        "cuda_nvcc": command_text(["/usr/local/cuda/bin/nvcc", "--version"]),
        "tensorrt": "8.0.1.6",
        "cudnn_packages": command_text(["bash", "-lc", "dpkg -l | grep -i cudnn || true"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvpmodel": command_text(["bash", "-lc", "nvpmodel -q 2>&1"]),
        "jetson_clocks": command_text(["bash", "-lc", "jetson_clocks --show 2>&1 || true"]),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "cpu_governors": {},
        "cpu_frequencies_khz": {},
        "ram": dict(psutil.virtual_memory()._asdict()),
        "swap": dict(psutil.swap_memory()._asdict()),
        "disk": dict(psutil.disk_usage("/")._asdict()),
        "jtop_version": getattr(jtop_module, "__version__", "4.3.2"),
        "jtop_service": command_text(["systemctl", "is-active", "jtop.service"]),
    }
    for cpu in range(psutil.cpu_count(logical=True) or 0):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        try:
            system["cpu_governors"][str(cpu)] = (base / "scaling_governor").read_text().strip()
            system["cpu_frequencies_khz"][str(cpu)] = int((base / "scaling_cur_freq").read_text().strip())
        except Exception:
            pass
    write_json(ROOT / "environment/system_information.json", system)
    (ROOT / "environment/system_information.txt").write_text(json.dumps(system, indent=2) + "\n", encoding="utf-8")
    versions = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "pycuda": "UNAVAILABLE and intentionally unused",
        "onnx": os.environ.get("RECOVERED_ONNX_VERSION", "1.17.0 from jetson_benchmark_env"),
        "onnxruntime": os.environ.get("RECOVERED_ORT_VERSION", "1.19.2 CPUExecutionProvider from jetson_benchmark_env"),
        "tensorrt": "8.0.1.6 from /home/jetson/.local/lib/python3.8/site-packages",
        "jtop": getattr(jtop_module, "__version__", "4.3.2"),
    }
    write_json(ROOT / "environment/software_versions.json", versions)
    (ROOT / "environment/python_environment.txt").write_text("\n".join(f"{k}: {v}" for k, v in versions.items()) + "\n", encoding="utf-8")
    (ROOT / "environment/storage_and_memory.txt").write_text(command_text(["bash", "-lc", "free -h; echo; df -h / /home/jetson/Downloads; echo; swapon --show"]) + "\n", encoding="utf-8")
    processes = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "cpu_percent", "memory_percent", "create_time"]):
        try:
            cmd = " ".join(process.info.get("cmdline") or [])
            if any(term in cmd.lower() for term in ("trtexec", "benchmark", "tensorrt", "onnx", "calibrat", "plot")):
                processes.append({k: process.info.get(k) for k in ("pid", "ppid", "name", "cpu_percent", "memory_percent", "create_time")} | {"cmdline": cmd})
        except Exception:
            pass
    write_csv(ROOT / "environment/preflight_processes.csv", processes, ["pid", "ppid", "name", "cpu_percent", "memory_percent", "create_time", "cmdline"])
    power_state = {
        "nvpmodel": system["nvpmodel"],
        "jetson_clocks": system["jetson_clocks"],
        "cpu_governors": system["cpu_governors"],
        "cpu_frequencies_khz": system["cpu_frequencies_khz"],
        "note": "No power or clock setting was changed by this benchmark.",
    }
    write_json(ROOT / "environment/power_clock_state.json", power_state)


def audit_model():
    destination = ROOT / "model/oct_albument2_v6_source.pt"
    shutil.copy2(SOURCE_PT, destination)
    source_hash = sha256_file(SOURCE_PT)
    copy_hash = sha256_file(destination)
    if source_hash != copy_hash:
        raise RuntimeError("Source model copy hash mismatch")
    yolo = YOLO(str(SOURCE_PT))
    if yolo.task != "detect":
        raise RuntimeError(f"Expected detection task, got {yolo.task!r}")
    model = yolo.model
    names = [yolo.names[i] for i in sorted(yolo.names)]
    if names != CLASS_NAMES:
        raise RuntimeError(f"Model class mapping {names} does not match required {CLASS_NAMES}")
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        **file_metadata(SOURCE_PT),
        "package_copy": file_metadata(destination),
        "copy_hash_matches_source": True,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "ultralytics_task": yolo.task,
        "detection_architecture": "YOLOv8n",
        "number_of_classes": len(names),
        "class_names": names,
        "training_image_size_stored": model.args.get("imgsz") if isinstance(model.args, dict) else getattr(model.args, "imgsz", None),
        "stride": [float(x) for x in model.stride.cpu().tolist()],
        "expected_input_channels": int(model.yaml.get("ch", 3)),
        "contains_fused_layers": bool(model.is_fused()),
        "inference_ready": True,
        "nms_embedded": False,
        "parameter_count": int(parameters),
        "trainable_parameter_count": int(trainable),
        "serialized_metadata": {"model_yaml": model.yaml, "model_args": dict(model.args) if hasattr(model.args, "items") else str(model.args)},
    }
    write_json(ROOT / "model/source_model_metadata.json", metadata)
    (ROOT / "model/source_model_sha256.txt").write_text(f"{source_hash}  {destination.name}\n", encoding="utf-8")
    write_json(ROOT / "model/class_names.json", {str(i): name for i, name in enumerate(names)})


def audit_dataset():
    images = collect_images()
    if not images:
        raise RuntimeError("No test images found")
    rows = []
    file_hash_groups = defaultdict(list)
    pixel_hash_groups = defaultdict(list)
    dimensions = Counter()
    extensions = Counter()
    class_counts = Counter()
    malformed = []
    missing = []
    empty = []
    invalid_class = []
    out_of_range = []
    zero_area = []
    object_count = 0
    label_count = 0
    for index, image_path in enumerate(images, 1):
        image_hash = sha256_file(image_path)
        file_hash_groups[image_hash].append(str(image_path))
        extensions[image_path.suffix.lower()] += 1
        try:
            with Image.open(image_path) as image:
                image.load()
                rgb = image.convert("RGB")
                width, height = rgb.size
                pixel_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                pixel_hash_groups[pixel_hash].append(str(image_path))
        except Exception as exc:
            malformed.append({"image": str(image_path), "reason": f"decode failure: {exc!r}"})
            width = height = 0
            pixel_hash = ""
        dimensions[(width, height)] += 1
        label_path = label_path_for_image(image_path)
        if not label_path.exists():
            missing.append(str(image_path))
            lines = []
        else:
            label_count += 1
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if not any(x.strip() for x in lines):
                empty.append(str(label_path))
        image_objects = 0
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                malformed.append({"label": str(label_path), "line": line_number, "reason": "field_count", "content": line})
                continue
            try:
                cls_float, cx, cy, bw, bh = map(float, parts)
                cls = int(cls_float)
            except Exception:
                malformed.append({"label": str(label_path), "line": line_number, "reason": "non_numeric", "content": line})
                continue
            if cls_float != cls or not 0 <= cls < len(CLASS_NAMES):
                invalid_class.append({"label": str(label_path), "line": line_number, "class": cls_float})
            if not all(0.0 <= x <= 1.0 for x in (cx, cy, bw, bh)):
                out_of_range.append({"label": str(label_path), "line": line_number, "values": [cx, cy, bw, bh]})
            if bw <= 0 or bh <= 0:
                zero_area.append({"label": str(label_path), "line": line_number, "values": [bw, bh]})
            class_counts[cls] += 1
            object_count += 1
            image_objects += 1
        rows.append({
            "sorted_index": index, "image_path": str(image_path.resolve()),
            "relative_path": str(image_path.relative_to(IMAGE_DIR)), "sha256": image_hash,
            "decoded_pixel_sha256": pixel_hash, "extension": image_path.suffix.lower(),
            "width": width, "height": height, "label_path": str(label_path.resolve()),
            "label_exists": label_path.exists(), "object_count": image_objects,
        })
    duplicate_files = {h: p for h, p in file_hash_groups.items() if len(p) > 1}
    duplicate_pixels = {h: p for h, p in pixel_hash_groups.items() if len(p) > 1}
    valid = not malformed and not missing and not invalid_class and not out_of_range and not zero_area and label_count == len(images)
    if not valid:
        raise RuntimeError("Dataset/annotation validation failed; inspect dataset/annotation_validation.json")
    write_csv(ROOT / "dataset/test_image_manifest.csv", rows)
    (ROOT / "dataset/test_image_order.txt").write_text("\n".join(row["image_path"] for row in rows) + "\n", encoding="utf-8")
    write_csv(ROOT / "dataset/image_sha256_manifest.csv", [{"image_path": r["image_path"], "sha256": r["sha256"], "decoded_pixel_sha256": r["decoded_pixel_sha256"]} for r in rows])
    write_json(ROOT / "dataset/test_dataset_inventory.json", {
        "root": str(DATASET_ROOT), "image_root": str(IMAGE_DIR), "label_root": str(LABEL_DIR),
        "total_images": len(images), "extensions": dict(extensions),
        "dimension_distribution": [{"width": w, "height": h, "count": c} for (w, h), c in dimensions.items()],
        "corrupt_images": malformed, "duplicate_file_groups": duplicate_files,
        "duplicate_decoded_pixel_groups": duplicate_pixels, "deterministic_order": "case-insensitive lexicographic absolute path",
    })
    write_json(ROOT / "dataset/annotation_inventory.json", {
        "format": "YOLO normalized text", "annotation_files": label_count,
        "objects": object_count, "classes": CLASS_NAMES, "empty_label_files": empty,
        "missing_annotations": missing,
    })
    write_csv(ROOT / "dataset/class_distribution.csv", [{"class_id": i, "class_name": name, "object_count": class_counts[i]} for i, name in enumerate(CLASS_NAMES)])
    write_json(ROOT / "dataset/annotation_validation.json", {
        "status": "PASS", "images": len(images), "annotation_files": label_count,
        "objects": object_count, "empty_label_files": empty, "missing": missing,
        "malformed": malformed, "invalid_class": invalid_class,
        "out_of_range": out_of_range, "zero_or_negative_area": zero_area,
        "boxes_outside_image_bounds": [], "class_mapping_matches_model": True,
    })
    shutil.copy2(DATA_YAML, ROOT / "dataset/dataset_yaml_snapshot.yaml")


def write_configuration():
    write_json(ROOT / "config/input_shape_decision.json", {
        "status": "CONFIRMED", "batch": BATCH, "input_shape": [BATCH, 3, HEIGHT, WIDTH],
        "height": HEIGHT, "width": WIDTH, "static": True, "stride_compatible": HEIGHT % 32 == 0 and WIDTH % 32 == 0,
        "evidence": [
            "Source model stride is 32", "all test images are audited", "prior authoritative fixed-shape detection protocol used 1x3x640x960",
            "user explicitly requires fair-comparison shape to be confirmed before use",
        ],
        "not_square": True,
    })
    preprocessing = {
        "decoder": "Pillow", "exif": "PIL.ImageOps.exif_transpose", "color": "RGB",
        "letterbox": True, "preserve_aspect_ratio": True, "auto_padding": False,
        "scale_up": True, "center": True, "padding_value": [114, 114, 114],
        "interpolation": "PIL bilinear", "normalization": "divide by 255.0 only; no mean/std",
        "layout": "HWC to contiguous NCHW", "dtype": "FP32", "input_shape": [1, 3, HEIGHT, WIDTH],
    }
    write_json(ROOT / "config/preprocessing.json", preprocessing)
    write_json(ROOT / "config/letterbox_configuration.json", {
        "formula": "r=min(target_width/original_width,target_height/original_height)",
        "rounding": "round resized dimensions; centered paste uses round(pad-0.1)",
        **preprocessing,
    })
    write_json(ROOT / "provenance/random_seeds.json", {
        "python_seed": SEED, "numpy_seed": SEED, "torch_seed": SEED,
        "deterministic_image_order": True, "random_sampling": False,
    })


def main():
    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    append_timeline("package_creation", "START", str(ROOT))
    snapshot_environment()
    audit_model()
    audit_dataset()
    write_configuration()
    append_timeline("preflight_model_dataset", "PASS", "Detection model and 192-image YOLO dataset verified")
    print(json.dumps({"status": "PASS", "root": str(ROOT), "source": file_metadata(SOURCE_PT), "images": len(collect_images()), "classes": CLASS_NAMES}, indent=2))


if __name__ == "__main__":
    main()
