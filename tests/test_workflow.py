from pathlib import Path
import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from data.dataset import build_dataset  # noqa: E402
from models.train import train_model, train_and_evaluate, TrainingConfig  # noqa: E402
from models.evaluate import (  # noqa: E402
    load_model,
    predict,
    json_success_rate,
)


def test_training_and_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "models.train.get_peft_model", lambda model, cfg: model
    )
    from transformers import AutoTokenizer as HFAT
    import models.train as train_mod

    orig = HFAT.from_pretrained

    def patched_tokenizer(name):
        tok = orig(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    monkeypatch.setattr(
        train_mod.AutoTokenizer, "from_pretrained", patched_tokenizer
    )
    from transformers import Trainer as HFTrainer

    monkeypatch.setattr(train_mod, "Trainer", HFTrainer)
    monkeypatch.setattr(HFTrainer, "train", lambda self: None)
    train_samples, val_samples = build_dataset(1, 1, seq_len=5, seed=0)
    cfg = TrainingConfig(
        model_name="sshleifer/tiny-gpt2",
        output_dir=str(tmp_path / "model"),
        epochs=1,
        batch_size=1,
        lr=1e-4,
    )
    train_model(train_samples, val_samples, cfg)
    tokenizer, model = load_model(cfg.output_dir)
    preds = predict(tokenizer, model, val_samples)
    assert len(preds) == len(val_samples)
    rate = json_success_rate(preds)
    assert 0.0 <= rate <= 1.0
    assert Path(cfg.output_dir).exists()

    # integrated helper
    rate2 = train_and_evaluate(train_samples, val_samples, cfg)
    assert 0.0 <= rate2 <= 1.0
