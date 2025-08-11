#!/usr/bin/env bash
# 读取 configs/gkd.yaml，调用 src/distill/gkd_trainer.py
# 关键: lambda/beta/seq_kd，student on-policy 生成再做KD

python -m src.distill.gkd_trainer --cfg configs/gkd.yaml "$@"
