# test8 Module

This module demonstrates a lightweight pipeline for building stock K-line datasets and training small language models to predict stock trends, provide investment advice and explain reasoning.

## Directory Structure

```
 test8/
 ├── config.py            # configuration of paths and hyper parameters
 ├── dataset_builder.py   # utilities for creating JSONL training data
 ├── data_loader.py       # wrapper around EastMoney K-line API
 ├── scripts/             # training and evaluation scripts
 ├── models/              # model checkpoints (created after training)
 └── run_all.sh           # example one-click command
```

## Dependencies

- Python >= 3.10
- pandas
- requests
- torch
- transformers

Install them with:

```bash
pip install -r requirements.txt
```

## Data Format

Each line in the training data is a JSON object with the keys used by the
training scripts:

```json
{
  "prompt": "<model input>",
  "label": "expected output"
}
```

## One-click Command

Run the complete workflow (dataset building and model training) with:

```bash
bash run_all.sh
```
