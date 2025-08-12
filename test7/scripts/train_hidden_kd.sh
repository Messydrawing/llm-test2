#!/usr/bin/env bash
# 原计划使用 accelerate + deepspeed 启动 DistillKit 的隐藏态蒸馏。
# 由于测试环境通常缺少这些依赖，我们改为直接调用一个轻量
# 的占位脚本以验证调用流程是否正常。

python distil_hidden.py --config configs/distill_hidden.yaml "$@"
