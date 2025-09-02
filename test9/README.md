# Test9 Project Skeleton

This directory provides a minimal scaffold for experimenting with supervised fine-tuning and reinforcement learning for language models.

## Structure
- **data/**: placeholders for training, validation, and test data.
- **configs/**: sample configuration files for SFT, PPO, and reward modeling.
- **src/**: module stubs for data processing, models, rewards, and evaluation utilities.
- **src/training/**: trainer stubs for SFT and PPO workflows.
- **scripts/**: entry points for training, inference, and evaluation.

## Data Generation
Run the helper below to download recent K-line data and build training,
validation and test splits:

```bash
python scripts/generate_data.py
```

The script pulls roughly 120 days of prices for the stock codes defined in
[`test4/config.py`](../test4/config.py). It extracts sliding 30‑day windows,
derives a simple up/down/stable label and writes three JSONL files under
`data/`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

Each line in these files has the form:

```json
{"prompt": "<30-day K-line data>", "completion": "{\"prediction\":...,\"analysis\":...,\"advice\":...}"}
```

## Usage
Populate the data and configuration files, implement the module contents, then
run the scripts to train and evaluate models.
