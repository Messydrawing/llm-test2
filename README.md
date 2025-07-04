# LLM Distillation Demo

This repository demonstrates a minimal end-to-end workflow for distilling a large
teacher model into a smaller student using LoRA fine-tuning. The code provides
placeholders for teacher model API calls and a simple Streamlit interface.

## Quickstart

1. Install the required Python packages. The repository includes a
   `requirements.txt` file with version pins for all libraries used in the
   demo:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit demo:
   ```bash
   streamlit run app.py
   ```

The demo will generate synthetic time series data, query the teacher model and
fine-tune a small student model using PEFT. After training the validation JSON
success rate is reported. By default the teacher model is mocked, but you can
enable real OpenAI calls by providing credentials.

### Synthetic data

`data.generate.generate_stock_series` creates synthetic stock price sequences
using a simple geometric Brownian motion. `data.dataset.build_dataset` applies
the teacher model to each series and returns two lists of samples:

```python
{"input": [1.0, ...], "label": {"market_type": "bullish", ...}}
```

The first list contains training samples and the second contains validation
samples.

### Command line usage

To train a model from the command line without the Streamlit UI run:

```bash
python models/train.py --num-train 4 --num-val 2 --seq-len 20 --epochs 1
```

Checkpoints are written to the directory specified by `--output-dir` (default:
`checkpoints`).

After training, evaluate the model:

```bash
python models/evaluate.py checkpoints --num-samples 2 --seq-len 20
```

Evaluation prints the JSON success rate and writes `preds.json` next to the
model files.

### Environment variables

`data/teacher.get_teacher_output` checks for the following variables:

- `OPENAI_API_KEY` – API key for OpenAI. When set, the teacher function will
  call the OpenAI chat completion API.
- `OPENAI_MODEL` – *(optional)* model name to use. Defaults to
  `gpt-3.5-turbo`.

Provide them as environment variables before running training or evaluation:

```bash
export OPENAI_API_KEY=sk-your-key
export OPENAI_MODEL=gpt-3.5-turbo   # optional
```

### Running tests

After installing the dependencies, execute the unit tests with `pytest`:

```bash
pytest
```

### End-to-end distillation

To run the full data collection, labeling, LoRA fine-tuning and evaluation
pipeline in one command execute:

```bash
python -m test1.distill
```

The script will write `labeled_data.jsonl`, `val_labeled_data.jsonl` and the
trained adapter under `lora_adapter`. Validation metrics are printed and saved
to `metrics.json`.

### `distill.py` command

`test1/distill.py` wraps dataset creation, teacher labeling, LoRA training and
evaluation. Run it from the repository root:

```bash
python -m test1.distill
```

Expected artifacts are:

- `labeled_data.jsonl` and `val_labeled_data.jsonl` – labeled datasets
- the fine-tuned adapter in the output directory (`lora_adapter` by default)
- `progress.png` inside the output directory showing metric trends
- `metrics.json` with evaluation scores

Set the `ARK_API_KEY` environment variable so the teacher model can label
samples.

Optional arguments allow customization:

```bash
python -m test1.distill --windows 3 --val-ratio 0.1 --out my_adapter
```

- `--windows` – number of windows per stock when building the dataset
- `--val-ratio` – fraction of data used for validation
- `--out` – directory for saving the adapter and progress plot
