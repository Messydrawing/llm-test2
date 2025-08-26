import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from test8.config import (
    BASE_MODEL_PATH,
    DATA_DIR,
    TREND_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None


class SFTDataset(Dataset):
    """Simple supervised fine-tuning dataset."""

    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int = 4096):
        with path.open("r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        prompt = example.get("prompt", "")
        output = example.get("label", "")
        if isinstance(output, dict):
            output = output.get("raw") or json.dumps(output, ensure_ascii=False)

        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=self.max_length
        ).input_ids
        output_ids = self.tokenizer(
            output + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids
        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_token_id: int):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = input_ids.ne(pad_token_id).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train trend model")
    parser.add_argument(
        "--train",
        type=Path,
        default=DATA_DIR / "train_trend.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=TREND_MODEL_PATH,
        help="Where to store the trained model",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory for training logs",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(BASE_MODEL_PATH),
        help="Base model identifier or path",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--use_8bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = getattr(args, "base_model", str(BASE_MODEL_PATH))
    data_path = Path(getattr(args, "train", DATA_DIR / "train_trend.jsonl"))
    output_path = Path(getattr(args, "model_out", TREND_MODEL_PATH))
    log_dir = getattr(args, "log_dir", None)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=args.use_8bit,
    )
    model.config.use_cache = False

    if args.use_lora and get_peft_model is not None:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    dataset = SFTDataset(data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        logging_dir=str(log_dir) if log_dir else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


if __name__ == "__main__":  # pragma: no cover
    main()
