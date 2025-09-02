"""Utility helpers for loading and saving language models.

These helpers are intentionally lightweight so that the unit tests can
exercise typical workflows without requiring heavyweight dependencies or
network access.  Each function degrades gracefully if optional packages are
missing.  The public API mirrors common patterns used in the other tests in
this repository.
"""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - optional dependencies
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - transformers not installed
    AutoModelForCausalLM = AutoTokenizer = object  # type: ignore


def load_tokenizer(model_name: str):
    """Load a :class:`~transformers.AutoTokenizer`.

    The tokenizer is loaded from ``model_name`` using
    :func:`~transformers.AutoTokenizer.from_pretrained`.  If the tokenizer does
    not define a pad token it is set to the EOS token to ensure compatibility
    with causal language modelling.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[attr-defined]
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, use_lora: bool = False):
    """Load an :class:`~transformers.AutoModelForCausalLM`.

    Parameters
    ----------
    model_name:
        Identifier passed to :func:`~transformers.AutoModelForCausalLM.from_pretrained`.
    use_lora:
        When ``True`` attempt to wrap the model with a PEFT LoRA adapter.  If
        PEFT is not available the function falls back to returning the base
        model.
    """

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[attr-defined]

    # Enable gradient checkpointing when supported to reduce memory usage
    if getattr(model, "gradient_checkpointing_enable", None):
        try:  # transformers >=4.29 added ``use_reentrant``
            model.gradient_checkpointing_enable(use_reentrant=False)
        except TypeError:  # pragma: no cover - older versions
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if use_lora:
        try:  # pragma: no cover - optional peft dependency
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
        except Exception:
            # If PEFT is unavailable we simply return the base model
            pass

    return model


def save_model(model, tokenizer, output_dir: str | Path) -> None:
    """Persist ``model`` and ``tokenizer`` to ``output_dir``.

    Any gradient-checkpointing hooks are disabled before saving so that the
    resulting checkpoint can be reloaded without requiring the training
    configuration.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Remove gradient checkpointing hooks if the model exposes the helper
    if getattr(model, "gradient_checkpointing_disable", None):
        model.gradient_checkpointing_disable()

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
