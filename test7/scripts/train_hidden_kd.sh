#!/usr/bin/env bash
# 用 accelerate + deepspeed 启动 DistillKit 的隐藏态蒸馏
# 关键变量: TEACHER/STUDENT/OUTPUT_DIR/EPOCHS/BF16/FA/DS_CONFIG
# 参考 DistillKit 官方 README 的启动方式

# TODO: 在此编写隐藏态KD的启动命令，例如：
# accelerate launch --config_file configs/distill_hidden.yaml distil_hidden.py
