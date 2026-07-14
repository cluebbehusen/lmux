"""Google structured output on Vertex AI: a responseJsonSchema built from a Pydantic model
returns content that validates back into that model, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lmux.types import ChatResponse, JsonSchemaResponseFormat, UserMessage
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 64
_PROMPT = "What is 17 times 23?"
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_structured_output.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


class MathAnswer(BaseModel):
    answer: int


_RESPONSE_FORMAT = JsonSchemaResponseFormat(name="math_answer", json_schema=MathAnswer.model_json_schema())


@pytest.mark.verified
def test_structured_output(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(
            _MODEL,
            [UserMessage(content=_PROMPT)],
            response_format=_RESPONSE_FORMAT,
            max_tokens=_MAX_TOKENS,
            provider_params=_NO_THINK,
        )

    resp = scenario(_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert_chat(resp, provider="google")
    assert resp.content is not None
    assert MathAnswer.model_validate_json(resp.content).answer == 391
    assert_cost(resp, **_RATES)
