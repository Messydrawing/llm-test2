#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(dirname "$SCRIPT_DIR")
MERGED_MODEL_DIR="merged_model"
OUTPUT_DIR="$BASE_DIR/models/$MERGED_MODEL_DIR"

mkdir -p "$OUTPUT_DIR"
cd "$BASE_DIR"

if command -v mergekit-moe-qwen2 >/dev/null 2>&1; then
  mergekit-moe-qwen2 scripts/merge_models_config.yml "$OUTPUT_DIR" --allow-crimes
else
  echo "mergekit-moe-qwen2 command not found, copying base model as fallback." >&2
  cp -r "$BASE_DIR/models/base_model/." "$OUTPUT_DIR" 2>/dev/null || true
fi

cp "$SCRIPT_DIR/configuration_qwen2.py" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/modeling_qwen2.py" "$OUTPUT_DIR/"
