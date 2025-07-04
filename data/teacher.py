"""Teacher model API interface."""

from typing import Any
import json
import os


def get_teacher_output(series: list[float]) -> dict[str, Any]:
    """Query the teacher model or return a mocked response.

    If ``OPENAI_API_KEY`` is set this function will attempt to query the OpenAI
    chat completion API using the model specified in ``OPENAI_MODEL`` (defaults
    to ``gpt-3.5-turbo``). The API is expected to return a JSON response which
    will be parsed and returned. If no key is configured or the API call fails,
    a simple mocked response is used.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai

            openai.api_key = api_key
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            prompt = (
                "Given the following price series, respond with a JSON object "
                "containing keys 'market_type', 'buy_time', 'trend_direction' "
                "and 'analysis'. Series: " + str(series)
            )
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception:
            pass

    return {
        "market_type": "bullish" if series[-1] > series[0] else "bearish",
        "buy_time": 0,
        "trend_direction": "up" if series[-1] > series[0] else "down",
        "analysis": "Mocked analysis based on synthetic data.",
    }
