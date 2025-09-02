"""Training utilities for the ``test9`` exercises.

The submodules provide light‑weight stand‑ins for the actual training helpers
used in the main project.  Only a very small public API is exported via
``__all__`` so that unit tests can import the pieces they need without pulling
in heavy dependencies.
"""

from .sft_trainer import SFTDataCollator, SFTTrainer
from .ppo_trainer import compute_rewards, generate_responses, ppo_step

__all__ = [
    "SFTDataCollator",
    "SFTTrainer",
    "compute_rewards",
    "generate_responses",
    "ppo_step",
]

