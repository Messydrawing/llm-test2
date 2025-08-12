#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."   # 切回 test7 根
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ACCEL_CFG="configs/accelerate_ds_zero3.yaml"
DISTILLKIT_ROOT="/zeng_gk/Zengxl/DistillKit-main"   # 你放 DistillKit 的目录
accelerate launch --config_file "$ACCEL_CFG" "$DISTILLKIT_ROOT/distil_hidden.py"
