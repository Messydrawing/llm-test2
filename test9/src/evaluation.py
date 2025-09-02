"""Evaluation utilities for JSON-based model outputs.

This module provides simple helpers to analyse model responses that are
expected to be JSON documents.  The main entry point
:func:`evaluate_json_output` validates a single string while
:func:`evaluate_dataset` aggregates metrics across many examples.

The implementation intentionally reuses helpers from :mod:`reward` to
avoid code duplication.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

from reward import is_valid_json, has_required_fields


def evaluate_json_output(output_str: str, reference_str: str | None = None) -> Dict[str, bool]:
    """Evaluate a single model output.

    Parameters
    ----------
    output_str:
        The raw string produced by the model.
    reference_str:
        Optional reference JSON string.  When provided, completeness is
        determined by checking that the model output contains at least
        the keys present in the reference.  Otherwise the
        :data:`reward.REQUIRED_KEYS` specification is used.

    Returns
    -------
    dict
        A dictionary with two boolean flags:

        ``{"is_valid": bool, "is_complete": bool}``
    """

    result = {"is_valid": False, "is_complete": False}
    if not isinstance(output_str, str):
        return result

    if not is_valid_json(output_str):
        return result

    result["is_valid"] = True
    obj = json.loads(output_str)

    if reference_str and is_valid_json(reference_str):
        ref_obj = json.loads(reference_str)
        result["is_complete"] = all(key in obj for key in ref_obj.keys())
    else:
        result["is_complete"] = has_required_fields(obj)

    return result


def evaluate_dataset(outputs: Sequence[str], references: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Aggregate evaluation metrics over a dataset.

    Parameters
    ----------
    outputs:
        Iterable of model output strings.
    references:
        Optional iterable of reference strings matching ``outputs`` in
        length.

    Returns
    -------
    dict
        Metrics summarising the dataset, currently consisting of
        ``json_validity_rate`` and ``json_completeness_rate``.
    """

    total = len(outputs)
    if total == 0:
        return {"json_validity_rate": 0.0, "json_completeness_rate": 0.0}

    valid = 0
    complete = 0

    for idx, output in enumerate(outputs):
        reference = references[idx] if references is not None else None
        res = evaluate_json_output(output, reference)
        if res["is_valid"]:
            valid += 1
        if res["is_complete"]:
            complete += 1

    return {
        "json_validity_rate": valid / total,
        "json_completeness_rate": complete / total,
    }

__all__ = ["evaluate_json_output", "evaluate_dataset"]
