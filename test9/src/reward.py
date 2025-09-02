"""Reward shaping utilities for PPO training.

This module provides a light‑weight reward function used during
reinforcement learning experiments. The behaviour is governed by a
JSON configuration file located at ``configs/reward_config.json``.
The values defined there can be tweaked without modifying the Python
code, making this a convenient spot for rapid experimentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "reward_config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG: Dict[str, Any] = json.load(f)
except FileNotFoundError:  # pragma: no cover - configuration missing in tests
    CONFIG = {}

# Configuration-driven list of mandatory keys expected in the model output.
REQUIRED_KEYS = CONFIG.get("required_keys", [])
# Phrases that should not appear in the output.
FORBIDDEN_PHRASES = CONFIG.get("forbidden_phrases", [])
# Target length range for model responses.
MIN_LENGTH = CONFIG.get("min_length", 0)
MAX_LENGTH = CONFIG.get("max_length", 10_000)

# Reward weights and penalties. Adjust these values to tune the reward model.
JSON_VALID_REWARD = 1.0
JSON_INVALID_PENALTY = 1.0
FIELD_REWARD = 1.0
FIELD_PENALTY = 1.0
LENGTH_BONUS = 0.5
LENGTH_PENALTY = 0.5
FORBIDDEN_PHRASE_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_valid_json(text: str) -> bool:
    """Return ``True`` if ``text`` is a valid JSON document.

    The function performs a fast validation using :func:`json.loads`. It is
    used to filter obviously malformed outputs before more expensive
    processing.
    """

    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def has_required_fields(obj: Dict[str, Any]) -> bool:
    """Check that all :data:`REQUIRED_KEYS` are present in ``obj``.

    Parameters
    ----------
    obj:
        Parsed JSON object to validate.
    """

    return all(key in obj for key in REQUIRED_KEYS)


# ---------------------------------------------------------------------------
# Reward calculation
# ---------------------------------------------------------------------------

def calculate_reward(output_str: str) -> float:
    """Compute a scalar reward for ``output_str``.

    The reward is composed of several simple, interpretable parts:

    * **JSON validity** – valid JSON earns :data:`JSON_VALID_REWARD`, otherwise
      :data:`JSON_INVALID_PENALTY` is subtracted.
    * **Required fields** – if the parsed object contains all
      :data:`REQUIRED_KEYS`, :data:`FIELD_REWARD` is added, otherwise
      :data:`FIELD_PENALTY` is subtracted.
    * **Length shaping** – responses within ``[MIN_LENGTH, MAX_LENGTH]`` gain
      :data:`LENGTH_BONUS`; otherwise :data:`LENGTH_PENALTY` is applied.
    * **Forbidden phrases** – every phrase in :data:`FORBIDDEN_PHRASES` found in
      the output incurs :data:`FORBIDDEN_PHRASE_PENALTY`.

    This design keeps the reward function easily adjustable for future tuning
    experiments.
    """

    reward = 0.0

    if is_valid_json(output_str):
        reward += JSON_VALID_REWARD
        obj = json.loads(output_str)
        if has_required_fields(obj):
            reward += FIELD_REWARD
        else:
            reward -= FIELD_PENALTY
    else:
        reward -= JSON_INVALID_PENALTY

    length = len(output_str)
    if MIN_LENGTH <= length <= MAX_LENGTH:
        reward += LENGTH_BONUS
    else:
        reward -= LENGTH_PENALTY

    lowered = output_str.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in lowered:
            reward -= FORBIDDEN_PHRASE_PENALTY

    return reward
