from pathlib import Path

# Base model path
BASE_MODEL_PATH = Path("./models/Qwen2.5-7B")

# Output paths for sub-models
TREND_MODEL_PATH = Path("./models/trend_model")
ADVICE_MODEL_PATH = Path("./models/advice_model")
EXPLANATION_MODEL_PATH = Path("./models/explanation_model")
MERGED_MODEL_PATH = Path("./models/MergedModel_7B")

# Data directory
DATA_DIR = Path("./data")

# Default training parameters
EPOCHS = 1
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Example stock codes used by dataset_builder
STOCK_CODES = ["600000"]
