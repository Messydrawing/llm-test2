import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import test6.inference as inf
from test6.teacher_labeler import label_samples


def test_call_gemini_mock(monkeypatch):
    class DummyResponse:
        text = "hello from gemini"

    class DummyClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = self

        def generate_content(self, model, contents):
            assert model == "gemini-2.5-flash"
            assert contents == "hi"
            return DummyResponse()

    monkeypatch.setattr(inf, "GEMINI_API_KEY", "x")
    # create fake google.genai module if not installed
    import types

    fake_mod = types.SimpleNamespace(Client=DummyClient)
    monkeypatch.setitem(sys.modules, "google.genai", fake_mod)

    res = inf.call_gemini("hi")
    assert res == {"content": "hello from gemini", "reasoning": ""}


def test_call_qwen_mock(monkeypatch):
    class DummyMessage:
        content = "qwen reply"

    class DummyResponse:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=DummyMessage())]

    class DummyChat:
        def __init__(self):
            self.completions = types.SimpleNamespace(
                create=lambda model, messages: DummyResponse()
            )

    class DummyClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = DummyChat()

    monkeypatch.setattr(inf, "DASHSCOPE_API_KEY", "token")
    import types

    fake_mod = types.SimpleNamespace(OpenAI=DummyClient)
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    res = inf.call_qwen("question")
    assert res == {"content": "qwen reply", "reasoning": ""}


def test_label_samples_writes_json(tmp_path):
    outputs = []

    def dummy(prompt):
        return {
            "content": prompt,
            "reasoning": f"why {prompt}",
        }

    out_file = tmp_path / "out.jsonl"
    records = label_samples(["a", "b"], out_file, call_teacher=dummy)
    assert len(records) == 2
    assert out_file.exists()
    loaded = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert loaded == records
    assert records[0]["label"] == {"raw": "a", "reasoning": "why a"}
    assert records[1]["label"] == {"raw": "b", "reasoning": "why b"}
