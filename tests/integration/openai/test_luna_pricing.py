"""OpenAI GPT-5.6 Luna Responses cost matches the published rate."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ResponseResponse
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-luna"
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "luna_pricing.json"
_RATES = {"input_rate": 0.20 / 1_000_000, "output_rate": 1.20 / 1_000_000}


def _respond(auth: Any, transport: Any) -> ResponseResponse:  # noqa: ANN401
    return OpenAIProvider(auth=auth, transport=transport).create_response(_MODEL, _PROMPT)


@pytest.mark.verified
def test_luna_pricing(scenario: Callable[..., Any], assert_cost: Callable[..., None]) -> None:
    response = scenario(_CASSETTE, _respond, requires="OPENAI_API_KEY")
    assert response.provider == "openai"
    assert "pong" in response.output_text.lower()
    assert_cost(response, **_RATES)
