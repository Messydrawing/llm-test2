#!/usr/bin/env python
"""Minimal PPO fine-tuning script.

The script loads a supervised fine-tuned checkpoint and further trains it
with a reward signal defined in ``src/reward.py`` using the ``trl``
``PPOTrainer``.  The implementation is intentionally compact so that unit
tests can execute quickly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from model_utils import load_model, load_tokenizer, save_model
from data_utils import load_dataset
from reward import calculate_reward

try:  # pragma: no cover - optional dependency
    from trl import PPOTrainer, PPOConfig
except Exception:  # pragma: no cover - trl not installed
    PPOTrainer = PPOConfig = None  # type: ignore


def train_ppo(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = Path(config_path).parent
    model_name = cfg["model_name"]
    sft_model_path = (cfg_dir / cfg.get("sft_model_path", "../models/sft")).resolve()
    prompt_path = (cfg_dir / cfg["prompt_path"]).resolve()
    output_dir = (cfg_dir / cfg.get("output_dir", "../models/ppo")).resolve()
    ppo_cfg = cfg.get("ppo_config", {})

    if PPOTrainer is None:  # pragma: no cover - handled when trl missing
        raise ImportError("trl library is required for PPO training")

    tokenizer = load_tokenizer(str(sft_model_path) if sft_model_path.exists() else model_name)
    model = load_model(str(sft_model_path) if sft_model_path.exists() else model_name)

    ppo_config = PPOConfig(**ppo_cfg)
    trainer = PPOTrainer(ppo_config, model, tokenizer)

    prompts = [ex["prompt"] for ex in load_dataset(str(prompt_path))]
    gen_kwargs = {"max_new_tokens": ppo_cfg.get("max_new_tokens", 32)}

    for prompt in prompts:
        query = tokenizer(prompt, return_tensors="pt")["input_ids"]
        response = trainer.generate(query, **gen_kwargs)
        text = tokenizer.decode(response[0], skip_special_tokens=True)
        reward = calculate_reward(text)
        trainer.step([query[0]], [response[0]], [reward])

    save_model(trainer.model, tokenizer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO fine-tuning")
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "ppo_config.yaml"
    parser.add_argument("--config", type=str, default=str(default_cfg))
    args = parser.parse_args()
    train_ppo(args.config)
