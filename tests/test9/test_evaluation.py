import sys
from pathlib import Path

import pytest

# Add test9/src to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[2] / 'test9' / 'src'))

from evaluation import evaluate_json_output, evaluate_dataset


def test_evaluate_json_output_valid_complete():
    output = '{"prediction": 1, "analysis": "good", "advice": "buy"}'
    result = evaluate_json_output(output)
    assert result == {"is_valid": True, "is_complete": True}


def test_evaluate_json_output_missing_fields():
    output = '{"prediction": 1}'
    result = evaluate_json_output(output)
    assert result["is_valid"] is True
    assert result["is_complete"] is False


def test_evaluate_json_output_invalid_json():
    result = evaluate_json_output('{bad json')
    assert result == {"is_valid": False, "is_complete": False}


def test_evaluate_json_output_reference():
    output = '{"a": 1, "b": 2}'
    reference = '{"a": 0, "b": 0}'
    assert evaluate_json_output(output, reference)["is_complete"] is True
    assert evaluate_json_output('{"a":1}', reference)["is_complete"] is False


def test_evaluate_dataset():
    outputs = [
        '{"prediction": 1, "analysis": "good", "advice": "buy"}',
        '{"prediction": 1}',
        '{bad json'
    ]
    metrics = evaluate_dataset(outputs)
    assert metrics["json_validity_rate"] == pytest.approx(2 / 3)
    assert metrics["json_completeness_rate"] == pytest.approx(1 / 3)
