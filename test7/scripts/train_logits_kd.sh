#!/usr/bin/env bash
# 原计划使用 accelerate 启动 DistillKit 的对数蒸馏。测试环境下
# 为了简化依赖，改为直接调用一个占位脚本。

python distil_logits.py --config configs/distill_logits.yaml "$@"
