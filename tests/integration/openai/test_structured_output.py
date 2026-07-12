"""OpenAI structured output: a json_schema response_format built from a Pydantic model
returns content that validates back into that model, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from lmux.types import ChatResponse, JsonSchemaResponseFormat, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 64
_PROMPT = "What is 17 times 23?"


class MathAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")  # emits additionalProperties: false, required by OpenAI strict mode

    answer: int


_RESPONSE_FORMAT = JsonSchemaResponseFormat(name="math_answer", json_schema=MathAnswer.model_json_schema(), strict=True)

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
    assert MathAnswer.model_validate_json(resp.content).answer == 391
    assert_cost(resp, **_RATES)
