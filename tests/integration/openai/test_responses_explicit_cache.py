"""OpenAI Responses accepts lmux explicit prompt-cache breakpoints."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ResponseInputMessage, ResponseResponse, TextContent
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-luna"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "responses_explicit_cache.json"


def _respond(auth: Any, transport: Any) -> ResponseResponse:  # noqa: ANN401
    input_items = [
        ResponseInputMessage(
            role="developer",
            content=[TextContent(text="Reply with exactly the word: pong"), CachePointContent()],
        )
    ]
    return OpenAIProvider(auth=auth, transport=transport).create_response(_MODEL, input_items)


@pytest.mark.verified
def test_responses_explicit_cache(scenario: Callable[..., Any]) -> None:
    response = scenario(_CASSETTE, _respond, requires="OPENAI_API_KEY")
    assert response.provider == "openai"
    assert "pong" in response.output_text.lower()
