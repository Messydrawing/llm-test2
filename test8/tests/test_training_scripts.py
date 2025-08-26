import sys
from pathlib import Path
import json
import argparse
import importlib

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))



def _dummy_tokenizer_cls():
    class DummyTokenizer:
        eos_token = "<eos>"
        pad_token = None
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, name, trust_remote_code=True):
            return cls()

        def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
            return type("Tokens", (), {"input_ids": [1]})

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    return DummyTokenizer


def _dummy_model_cls():
    class DummyModel:
        config = type("Cfg", (), {"use_cache": False})

        @classmethod
        def from_pretrained(cls, name, trust_remote_code=True, device_map=None, load_in_8bit=False):
            return cls()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    return DummyModel


def _dummy_trainer_cls():
    class DummyTrainer:
        def __init__(self, model, args, train_dataset, data_collator):
            pass

        def train(self):
            pass

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    return DummyTrainer


# ---------------------------------------------------------------------------


import pytest


@pytest.mark.parametrize(
    "mod_name,data_file,path_attr",
    [
        ("test8.scripts.train_trend_model", "train_trend.jsonl", "TREND_MODEL_PATH"),
        ("test8.scripts.train_advice_model", "train_advice.jsonl", "ADVICE_MODEL_PATH"),
        (
            "test8.scripts.train_explanation_model",
            "train_explain.jsonl",
            "EXPLANATION_MODEL_PATH",
        ),
    ],
)
def test_training_scripts_load(monkeypatch, tmp_path, mod_name, data_file, path_attr):
    mod = importlib.import_module(mod_name)
    data_path = tmp_path / data_file
    data_path.write_text(
        json.dumps({"prompt": "hi", "label": {"raw": "bye"}}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "BASE_MODEL_PATH", "dummy-model")
    monkeypatch.setattr(mod, path_attr, tmp_path / "out")

    monkeypatch.setattr(mod, "AutoTokenizer", _dummy_tokenizer_cls())
    monkeypatch.setattr(mod, "AutoModelForCausalLM", _dummy_model_cls())
    monkeypatch.setattr(mod, "Trainer", _dummy_trainer_cls())
    monkeypatch.setattr(mod, "collate_fn", lambda batch, pad_token_id: {})
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            epochs=1,
            learning_rate=0.0,
            batch_size=1,
            gradient_accumulation_steps=1,
            use_lora=False,
            lora_r=0,
            lora_alpha=0,
            lora_dropout=0.0,
            use_8bit=False,
        ),
    )

    mod.main()
