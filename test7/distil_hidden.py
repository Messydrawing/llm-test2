#!/usr/bin/env python
"""Lightweight placeholder for DistillKit's distil_hidden.py.
Reads a YAML config and prints a short summary. This allows the
training pipeline to run in environments where the real DistillKit
package is not available."""
import argparse
from pathlib import Path
import yaml

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock hidden-state distillation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    teacher = cfg.get("teacher", "unknown-teacher")
    student = cfg.get("student", "unknown-student")
    print(f"[distil_hidden] {teacher} -> {student}")

    out_dir = cfg.get("output_dir")
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoint.txt").write_text("placeholder checkpoint\n", encoding="utf-8")

if __name__ == "__main__":
    main()
