#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil

from common import ROOT, append_timeline, write_csv, write_json
from preprocessing import TIMING_FIELDS


def repair(path, backup):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    values = [float(row["preprocessing_total_ms"]) for row in rows]
    if min(values) < 1_000_000.0:
        raise RuntimeError(f"Refusing ambiguous or already-corrected input: {path}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    fields = list(rows[0].keys())
    for row in rows:
        for field in TIMING_FIELDS:
            row[field] = repr(float(row[field]) * 1e-9)
    write_csv(path, rows, fields)
    return len(rows), min(values), max(values)


def main():
    marker = ROOT / "validation/timing_unit_repair.json"
    if marker.exists():
        raise FileExistsError("Timing-unit repair marker already exists; refusing repeated conversion")
    append_timeline("timing_unit_repair", "START", "offline exact unit conversion; zero inference")
    official = ROOT / "raw/per_image_timings.csv"
    warmup = ROOT / "raw/warmup_timings.csv"
    backup_dir = ROOT / "raw/pre_timing_unit_correction"
    official_info = repair(official, backup_dir / official.name)
    warmup_info = repair(warmup, backup_dir / warmup.name)
    evidence = {
        "status": "PASS",
        "no_inference_executed": True,
        "affected_fields": TIMING_FIELDS,
        "correction": "stored value multiplied by 1e-9; original code used (end_ns-start_ns)*1000 while correct milliseconds are (end_ns-start_ns)/1e6",
        "unaffected": ["H2D", "TensorRT engine", "D2H", "transfer-inclusive", "postprocessing", "NMS", "end-to-end", "wall-clock", "telemetry", "energy", "predictions", "metrics"],
        "official_rows": official_info[0],
        "warmup_rows": warmup_info[0],
        "original_official_preprocessing_value_range": [official_info[1], official_info[2]],
        "original_warmup_preprocessing_value_range": [warmup_info[1], warmup_info[2]],
        "backups": [str(backup_dir / official.name), str(backup_dir / warmup.name)],
    }
    write_json(marker, evidence)
    append_timeline("timing_unit_repair", "PASS", "192 official + 10 warmup rows corrected; zero inference")
    print(evidence)


if __name__ == "__main__":
    main()
