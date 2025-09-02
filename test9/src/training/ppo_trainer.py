"""Helper wrappers around :class:`trl.PPOTrainer`.

The real project performs reinforcement learning using the `trl` package.  In
the exercises found in this repository we only need small convenience
functions to glue together generation, reward computation and the PPO update
step.  These wrappers keep the actual training loop in the tests concise while
remaining importable even if optional dependencies are absent.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from trl import PPOTrainer
except Exception:  # pragma: no cover - TRL not installed
    PPOTrainer = object  # type: ignore

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------


def generate_responses(
    trainer: PPOTrainer, prompts: Sequence[str], generation_kwargs: dict | None = None
) -> List[str]:
    """Generate model responses for ``prompts``.

    The helper delegates to ``trainer.model.generate`` and decodes the output
    using ``trainer.tokenizer``.  Only the small subset of arguments required
    in the tests is supported.  When :mod:`torch` is unavailable the function
    falls back to a minimal implementation returning raw token IDs.
    """

    if getattr(trainer, "model", None) is None or getattr(trainer, "tokenizer", None) is None:
        raise RuntimeError("trainer must expose 'model' and 'tokenizer' attributes")

    gen_kwargs = generation_kwargs or {}
    tokenised = trainer.tokenizer(list(prompts), return_tensors="pt", padding=True)
    input_ids = tokenised["input_ids"]
    if torch is not None and hasattr(trainer.model, "device"):
        input_ids = input_ids.to(trainer.model.device)

    output_ids = trainer.model.generate(input_ids=input_ids, **gen_kwargs)

    if torch is not None:
        return trainer.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # Fallback – return raw token ids to keep behaviour predictable
    return [ids for ids in output_ids.tolist()]


# ---------------------------------------------------------------------------
# Reward calculation
# ---------------------------------------------------------------------------


def compute_rewards(reward_fn: Callable[[str], float], responses: Iterable[str]) -> List[float]:
    """Evaluate ``reward_fn`` for each response."""

    return [float(reward_fn(resp)) for resp in responses]


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------


def ppo_step(
    trainer: PPOTrainer,
    prompts: Sequence[str],
    reward_fn: Callable[[str], float],
    generation_kwargs: dict | None = None,
) -> Tuple[List[str], List[float]]:
    """Run a single PPO optimisation step.

    Parameters
    ----------
    trainer:
        Instance of :class:`trl.PPOTrainer`.
    prompts:
        Sequence of input prompts fed to the model.
    reward_fn:
        Callable computing a reward from the generated text.
    generation_kwargs:
        Optional keyword arguments forwarded to
        :meth:`transformers.PreTrainedModel.generate`.

    Returns
    -------
    Tuple[List[str], List[float]]
        The generated responses and the corresponding rewards.
    """

    responses = generate_responses(trainer, prompts, generation_kwargs)
    rewards = compute_rewards(reward_fn, responses)

    # The PPO trainer expects tensors of token IDs.  For the purposes of the
    # unit tests the prompts and responses are fed in as raw strings which the
    # trainer knows how to tokenise internally.
    trainer.step(prompts, responses, rewards)

    return responses, rewards


__all__ = ["generate_responses", "compute_rewards", "ppo_step"]

