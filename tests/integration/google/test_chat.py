"""Google chat on Vertex AI (via API key): gemini-2.5-flash returns the expected word,
and cost matches the published global-endpoint rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})  # thinking off -> deterministic short output
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_chat.json"

# gemini-2.5-flash published rates ($/token, global endpoint).
_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


@pytest.mark.verified
def test_chat(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(
            _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
        )

    resp = scenario(_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert_chat(resp, provider="google")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
