#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_DIR=${1:-"$REPO_DIR/models"}
MERGED_MODEL_DIR="merged_model"
OUTPUT_DIR=${2:-"$MODEL_DIR/$MERGED_MODEL_DIR"}

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

if command -v mergekit-moe-qwen2 >/dev/null 2>&1; then
  mergekit-moe-qwen2 test8/scripts/merge_models_config.yml "$OUTPUT_DIR" --allow-crimes
else
  echo "mergekit-moe-qwen2 command not found, copying base model as fallback." >&2
  cp -r "$MODEL_DIR/base_model/." "$OUTPUT_DIR" 2>/dev/null || true
fi

cp "$SCRIPT_DIR/configuration_qwen2.py" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/modeling_qwen2.py" "$OUTPUT_DIR/"
