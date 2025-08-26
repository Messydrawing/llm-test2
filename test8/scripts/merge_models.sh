#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(dirname "$SCRIPT_DIR")
OUTPUT_DIR="$BASE_DIR/models/merged_model"

mkdir -p "$OUTPUT_DIR"
cd "$BASE_DIR"

mergekit-moe-qwen2 scripts/merge_models_config.yml "$OUTPUT_DIR" --allow-crimes

cp "$SCRIPT_DIR/configuration_qwen2.py" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/modeling_qwen2.py" "$OUTPUT_DIR/"
