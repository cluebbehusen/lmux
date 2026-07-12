"""OpenAI structured output: a json_schema response_format returns content that
parses to the expected answer, and cost matches the published rate.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, JsonSchemaResponseFormat, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 64
_PROMPT = "What is 17 times 23?"
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT = JsonSchemaResponseFormat(name="math_answer", json_schema=_SCHEMA, strict=True)

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "structured_output.json"

# gpt-4o-mini published rates ($/token).
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return OpenAIProvider(auth=auth, transport=transport).chat(
        _MODEL, [UserMessage(content=_PROMPT)], response_format=_RESPONSE_FORMAT, max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_structured_output(
    scenario: Callable[..., Any], assert_chat: Callable[..., None], assert_cost: Callable[..., None]
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="OPENAI_API_KEY")
    assert_chat(resp, provider="openai")
    assert resp.content is not None
    assert json.loads(resp.content)["answer"] == 391
    assert_cost(resp, **_RATES)
