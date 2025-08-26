#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# directories for data, labels, models, logs
DATA_DIR="$SCRIPT_DIR/data/processed"
LABEL_DIR="$SCRIPT_DIR/data/labeled"
MODEL_DIR="$SCRIPT_DIR/models"
LOG_DIR="$SCRIPT_DIR/logs"
BASE_MODEL="$MODEL_DIR/base_model"  # local base model path

mkdir -p "$DATA_DIR" "$LABEL_DIR" "$MODEL_DIR" "$LOG_DIR"

cd "$REPO_DIR"

# 1. Build dataset
python -m test8.dataset_builder --out-dir "$DATA_DIR"

# 2. Teacher labeling
python -m test8.teacher_labeler --data "$DATA_DIR/train.jsonl" --out-dir "$LABEL_DIR"

# 3. Train models
python -m test8.train_trend_model --train "$LABEL_DIR/train_trend.jsonl" --model-out "$MODEL_DIR/trend" --log-dir "$LOG_DIR/trend" --base-model "$BASE_MODEL"
python -m test8.train_advice_model --train "$LABEL_DIR/train_advice.jsonl" --model-out "$MODEL_DIR/advice" --log-dir "$LOG_DIR/advice" --base-model "$BASE_MODEL"
python -m test8.train_explanation_model --train "$LABEL_DIR/train_explain.jsonl" --model-out "$MODEL_DIR/explain" --log-dir "$LOG_DIR/explain" --base-model "$BASE_MODEL"

# 4. Merge models
bash "$SCRIPT_DIR/merge_models.sh" "$MODEL_DIR" "$MODEL_DIR/merged"

# 5. Evaluate merged model
python -m test8.evaluate_models --data "$DATA_DIR/val.jsonl" --model "$MODEL_DIR/merged" --log-dir "$LOG_DIR"
