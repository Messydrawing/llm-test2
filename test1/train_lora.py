# test1/train_lora.py  ── v4.1 (prompt‑safe, log‑clean, metrics‑ready) ─────────────
from __future__ import annotations

import argparse, json, os, logging, warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ─── 1 ▸ 静音 transformers 的 INFO/WARNING ─────────────────────────────
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()                               # 只保留 ERROR
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Caching is incompatible")

# ─── 2 ▸ CUDA/TF32 & BnB 内存设置 ─────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
)

# ╭─────────────────────  配置  ─────────────────────╮
@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen1.5-7B"
    data_path: str = "labeled_data.jsonl"
    eval_path: str | None = None
    output_dir: str = "lora_adapter"
    batch_size: int = 1
    lr: float = 2e-4
    epochs: int | None = None
    max_steps: int | None = None
    grad_accum: int = 4
    max_len: int = 1024            # prompt+label 截断上限

# ╭──────────────────── 数据加载 ────────────────────╮
def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(x) for x in f]
    return Dataset.from_list(records)

# ╭──────────────────── Collator ────────────────────╮
class LabelCollator:
    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch):
        texts, prompt_lens = [], []
        for rec in batch:
            prompt = rec["prompt"].strip()
            raw_label = rec["label"]
            label = (
                json.dumps(raw_label, ensure_ascii=False, sort_keys=True).strip()
                if isinstance(raw_label, (dict, list))
                else str(raw_label).strip()
            )
            full = f"{prompt}\n\n### 答案：{label}"
            plen = len(self.tok(prompt + "\n\n### 答案：")["input_ids"])
            prompt_lens.append(plen)
            texts.append(full)

        model_inputs = self.tok(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        if (labels != -100).sum() == 0:
            raise ValueError(
                "全部 label 被截掉；请缩短 prompt 或增大 --max-len。"
            )

        model_inputs["labels"] = labels
        return model_inputs

# ╭──────────────────── 评估 & 画图 ─────────────────╮
def _eval_model(tok, mdl, ps, rs) -> dict[str, float]:
    from .inference import call_student
    from . import evaluate
    preds = [call_student(tok, mdl, p) for p in ps]
    return {
        "bleu": evaluate.bleu_score(rs, preds),
        "rougeL": evaluate.rouge_l(rs, preds),
        "embed": evaluate.embedding_similarity(rs, preds),
    }

def _plot_scores(scores, path):
    if not scores:
        return
    epochs = list(range(1, len(scores) + 1))
    fig, ax = plt.subplots()
    for k in scores[0]:
        ax.plot(epochs, [s[k] for s in scores], label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

class EvalCallback(TrainerCallback):
    def __init__(self, tok, ds_eval: Dataset | None, out_dir: str):
        self.tok = tok
        if ds_eval:
            self.prompts = [x["prompt"] for x in ds_eval]
            self.refs = [
                json.dumps(x["label"], ensure_ascii=False, sort_keys=True)
                if isinstance(x["label"], dict)
                else str(x["label"])
                for x in ds_eval
            ]
        else:
            self.prompts, self.refs = [], []
        self.scores, self.fig = [], os.path.join(out_dir, "progress.png")

    def on_epoch_end(self, args, state, control, **kw):
        if not self.prompts:
            return
        metrics = _eval_model(self.tok, kw["model"], self.prompts, self.refs)
        self.scores.append(metrics)
        _plot_scores(self.scores, self.fig)

# ╭──────────────────── 主函数 ────────────────────╮
def main(cfg: TrainConfig) -> None:
    ds_train = load_dataset(cfg.data_path)
    ds_eval = load_dataset(cfg.eval_path) if cfg.eval_path else None
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map="auto",
        quantization_config=bnb, trust_remote_code=True
    )
    try:
        base.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        base.gradient_checkpointing_enable()
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        )
    )
    model.print_trainable_parameters()

    max_steps = cfg.max_steps if cfg.max_steps is not None else (-1 if cfg.epochs else 200)

    try:          # transformers ≥4.22
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs or 1,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=1,
            evaluation_strategy="epoch" if ds_eval else "no",
            save_strategy="epoch",
            remove_unused_columns=False,
        )
    except TypeError:  # older transformers
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs or 1,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=1,
            eval_strategy="epoch" if ds_eval else "no",
            save_strategy="epoch",
            remove_unused_columns=False,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=LabelCollator(tok, cfg.max_len),
    )
    if ds_eval:
        trainer.add_callback(EvalCallback(tok, ds_eval, cfg.output_dir))
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    if ds_eval:
        print(f"[✓] progress figure → {cfg.output_dir}/progress.png")

# ╭──────────────────── CLI ────────────────────╮
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LoRA fine-tune (safe)")
    ap.add_argument("--data", default="labeled_data.jsonl")
    ap.add_argument("--eval-data")
    ap.add_argument("--base-model", default="Qwen/Qwen1.5-7B")
    ap.add_argument("--out", default="lora_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        eval_path=args.eval_data,
        base_model=args.base_model,
        output_dir=args.out,
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_len=args.max_len,
    )
    main(cfg)
