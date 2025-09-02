import json
import sys
from pathlib import Path

# Add test9/src to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'test9' / 'src'))

from data_utils import load_dataset, preprocess_example, make_datasets


class DummyTokenizer:
    """A minimal whitespace tokenizer for testing."""

    def __init__(self):
        self.vocab = {}
        self.next_id = 1
        self.eos_token_id = 0

    def _tokenize(self, text):
        tokens = text.strip().split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[tok])
        return ids

    def __call__(self, text, add_special_tokens=True):
        ids = self._tokenize(text)
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids}


def test_load_dataset(tmp_path):
    path = tmp_path / 'data.jsonl'
    examples = [
        {"prompt": "hello", "completion": " world"},
        {"prompt": "foo", "completion": " bar"},
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    data = load_dataset(str(path))
    assert isinstance(data, list)
    assert data == examples


def test_preprocess_example_masking():
    tokenizer = DummyTokenizer()
    example = {"prompt": "hello", "completion": " world"}
    out = preprocess_example(example, tokenizer)
    assert out["input_ids"] == [1, 2, 0]
    assert out["labels"] == [-100, 2, 0]


def test_make_datasets(tmp_path):
    tokenizer = DummyTokenizer()
    train_path = tmp_path / 'train.jsonl'
    val_path = tmp_path / 'val.jsonl'
    data = {"prompt": "hi", "completion": " there"}
    for p in (train_path, val_path):
        with open(p, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')

    train_ds, val_ds = make_datasets(str(train_path), str(val_path), tokenizer)
    assert len(train_ds) == 1
    assert train_ds[0]["labels"][0] == -100
    assert len(val_ds) == 1
