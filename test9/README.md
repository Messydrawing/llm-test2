# Test9 Project

## Overview

`test9` contains a compact end-to-end example for reinforcement learning from
human feedback (RLHF) built around stock price movement data.  The workflow
covers:

1. generating labelled training data,
2. supervised fine-tuning (SFT) of a base language model,
3. reinforcement learning with Proximal Policy Optimization (PPO),
4. inference and evaluation of the resulting model.

The code is intentionally lightweight so that each stage can be run on a
single GPU or even CPU for very small experiments.

## Directory Map

```
test9/
├── configs/   # YAML and JSON configs for training and reward shaping
├── data/      # Generated train/val/test datasets
├── scripts/   # Command line entry points for the workflow
├── src/       # Library code (data, models, reward, evaluation, training)
└── README.md  # This file
```

Trained checkpoints are written to a `models/` directory when running the
training scripts.

## Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/) with CUDA support for GPU training
- `transformers`, `datasets`, `trl`, and `peft` libraries
- At least 12 GB of GPU memory is recommended for full training runs; the code
  will fall back to CPU where possible for quick tests.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # run from repository root
```

## Data Generation

Download recent K-line price data and build the train/validation/test splits:

```bash
python scripts/generate_data.py
```

The helper pulls roughly 120 days of prices for the stock codes defined in
[`test4/config.py`](../test4/config.py).  It extracts sliding 30‑day windows,
derives a simple up/down/stable label and writes three JSONL files under
`data/`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

Each line has the format:

```json
{"prompt": "<30-day K-line data>", "completion": "{\"prediction\":...,\"analysis\":...,\"advice\":...}"}
```

## Supervised Fine-Tuning (SFT)

Train a base model on the generated data:

```bash
python scripts/train_sft.py --config configs/sft_config.yaml --output_dir models/sft
```

The YAML config selects the model, data files and `transformers` training
arguments.  The resulting checkpoint is saved under `models/sft`.

## PPO Training

Further optimise the SFT model with a reward function:

```bash
python scripts/train_ppo.py --config configs/ppo_config.yaml
```

The script loads the SFT checkpoint, generates responses for prompts in
`data/train.jsonl`, scores them with `src/reward.py` and performs PPO updates.
The fine-tuned model is saved to `models/ppo` by default.

## Inference

Generate predictions with the PPO model:

```bash
python scripts/infer.py --model-path models/ppo "<your prompt here>"
# or read prompts from a file
python scripts/infer.py --model-path models/ppo --input-file prompts.txt
```

Outputs are printed as JSON objects on stdout.

## Evaluation

Score model outputs on the held-out test set:

```bash
python scripts/evaluate.py --model-path models/ppo --test-path data/test.jsonl
```

The script validates JSON format and required fields, reporting aggregate
statistics such as accuracy of required keys.

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  (Schulman et al., 2017)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  (Hu et al., 2021)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
  (Ouyang et al., 2022)
- [TRL – Transformer Reinforcement Learning library](https://github.com/huggingface/trl)

These resources informed the overall design and algorithms used in this
project.

