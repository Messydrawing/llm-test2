import argparse
import json
from dataclasses import dataclass
from typing import Any
import os

import matplotlib.pyplot as plt

from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

from . import evaluate
from .inference import call_student


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen1.5-7B"
    data_path: str = "labeled_data.jsonl"
    eval_path: str | None = None
    output_dir: str = "lora_adapter"
    batch_size: int = 1
    lr: float = 1e-4
    epochs: int | None = None
    max_steps: int | None = None
    grad_accum: int = 4


def load_dataset(path: str) -> Dataset:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


class LabelCollator:
    def __init__(self, tokenizer, answer_prefix: str = "\n回答："):
        self.tok = tokenizer
        self.prefix = answer_prefix
        # 避免某些模型没显式 pad_token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch):
        import torch

        prompts = [b["prompt"].strip() for b in batch]
        answers = [
            json.dumps(b["label"], ensure_ascii=False).strip() for b in batch
        ]

        # 1️  先把「提示+答案」拼成完整输入
        sources = [p + self.prefix + a for p, a in zip(prompts, answers)]
        model_inputs = self.tok(
            sources,
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        # 2️  再为 labels 制作掩码：提示部分 = -100，答案部分 = token_id
        #    这样 loss 只在答案位置计算
        labels = model_inputs["input_ids"].clone()
        for i, p in enumerate(prompts):
            prompt_len = len(
                self.tok(p + self.prefix, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            labels[i, :prompt_len] = -100
        model_inputs["labels"] = labels
        return model_inputs


def _eval_model(tokenizer, model, prompts, refs) -> dict[str, float]:
    preds = [call_student(tokenizer, model, p) for p in prompts]
    return {
        "bleu": evaluate.bleu_score(refs, preds),
        "rougeL": evaluate.rouge_l(refs, preds),
        "embed": evaluate.embedding_similarity(refs, preds),
    }


def _plot_scores(scores: list[dict[str, float]], path: str) -> None:
    epochs = list(range(1, len(scores) + 1))
    fig, ax = plt.subplots()
    for metric in scores[0].keys():
        ax.plot(epochs, [s[metric] for s in scores], label=metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, out_dir: str):
        self.tok = tokenizer
        self.prompts, self.refs = [], []
        if eval_dataset is not None:
            for rec in eval_dataset:
                self.prompts.append(rec["prompt"])
                label = rec.get("label", "")
                if isinstance(label, dict):
                    self.refs.append(json.dumps(label, ensure_ascii=False))
                else:
                    self.refs.append(str(label))
        self.out_dir = out_dir
        self.scores: list[dict[str, float]] = []
        self.fig_path = os.path.join(out_dir, "progress.png")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.prompts:
            return
        model = kwargs.get("model")
        metrics = _eval_model(self.tok, model, self.prompts, self.refs)
        self.scores.append(metrics)
        _plot_scores(self.scores, self.fig_path)


def main(cfg: TrainConfig) -> None:
    ds = load_dataset(cfg.data_path)
    eval_ds = load_dataset(cfg.eval_path) if cfg.eval_path else None
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(use_reentrant=False)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(model, peft_cfg)

    max_steps = cfg.max_steps
    if cfg.epochs is None:
        max_steps = max_steps or 200
    elif max_steps is None:
        max_steps = -1

    try:
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs if cfg.epochs is not None else 1,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=10,
            evaluation_strategy="epoch" if cfg.eval_path else "no",
            save_strategy="epoch",
            remove_unused_columns=False,
        )
    except TypeError:
        # Older ``transformers`` versions used ``eval_strategy`` instead of
        # ``evaluation_strategy``. Fall back to the legacy argument for
        # backward compatibility.
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs if cfg.epochs is not None else 1,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=10,
            eval_strategy="epoch" if cfg.eval_path else "no",
            save_strategy="epoch",
            remove_unused_columns=False,
        )

    collator = LabelCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    if eval_ds is not None:
        trainer.add_callback(EvalCallback(tokenizer, eval_ds, cfg.output_dir))
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if eval_ds is not None:
        print(
            f"Progress figure saved to: {os.path.join(cfg.output_dir, 'progress.png')}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument(
        "--data", default="labeled_data.jsonl", help="Dataset path"
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen1.5-7B", help="Base model"
    )
    parser.add_argument("--out", default="lora_adapter", help="Output dir")
    parser.add_argument("--eval-data", help="Validation dataset path")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Training epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps (overrides epochs if set)",
    )
    args = parser.parse_args()
    cfg = TrainConfig(
        data_path=args.data,
        eval_path=args.eval_data,
        base_model=args.base_model,
        output_dir=args.out,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    main(cfg)
