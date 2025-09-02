import json
from typing import List, Dict, Any, Tuple

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - datasets not installed
    Dataset = None  # type: ignore

def load_dataset(path: str):
    """Load a JSONL dataset.

    Each line of the file at ``path`` should contain a JSON object with at
    least ``prompt`` and ``completion`` fields. The function returns a list of
    dictionaries representing the examples. If the optional :mod:`datasets`
    package is available, the caller may convert the list to a
    :class:`datasets.Dataset` via ``Dataset.from_list``.
    """
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def preprocess_example(example: Dict[str, Any], tokenizer):
    """Tokenize an example for causal language modeling.

    The ``example`` must provide ``prompt`` and ``completion`` strings. These
    are concatenated and tokenized with ``tokenizer``. Tokens corresponding to
    the prompt are masked in the ``labels`` field with ``-100`` so that only
    completion tokens contribute to the loss during training.
    """
    prompt = example["prompt"]
    completion = example["completion"]
    text = prompt + completion

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(text, add_special_tokens=True)
    labels = tokenized["input_ids"].copy()
    prompt_len = len(prompt_tokens)
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

def make_datasets(train_path: str, val_path: str, tokenizer) -> Tuple[Dataset, Dataset]:
    """Load and preprocess training and validation datasets.

    ``train_path`` and ``val_path`` should point to JSONL files formatted for
    :func:`load_dataset`. The returned datasets contain tokenized examples
    suitable for training causal language models.
    """
    if Dataset is None:  # pragma: no cover - handled in environments without datasets
        raise ImportError("datasets library is required for make_datasets")

    train_raw = load_dataset(train_path)
    val_raw = load_dataset(val_path)

    train_ds = Dataset.from_list([preprocess_example(ex, tokenizer) for ex in train_raw])
    val_ds = Dataset.from_list([preprocess_example(ex, tokenizer) for ex in val_raw])
    return train_ds, val_ds
