"""OpenAI Responses API: ``create_response`` returns text output with an id, and cost matches the published rate."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ResponseResponse
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-4o-mini"
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "responses.json"

# gpt-4o-mini published rates ($/token).
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


def _respond(auth: Any, transport: Any) -> ResponseResponse:  # noqa: ANN401 — harness-supplied per mode
    return OpenAIProvider(auth=auth, transport=transport).create_response(_MODEL, _PROMPT)


@pytest.mark.verified
def test_responses(scenario: Callable[..., Any], assert_cost: Callable[..., None]) -> None:
    resp = scenario(_CASSETTE, _respond, requires="OPENAI_API_KEY")
    assert resp.provider == "openai"
    assert isinstance(resp.output_text, str)
    assert resp.id
    assert_cost(resp, **_RATES)
