"""Google chat on the Gemini Developer API (via API key): a generateContent call over the
v1beta path returns a mapped response with cost — the Dev-API counterpart to the Vertex path.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage

_MODEL = "gemini-3-flash-preview"
_MAX_TOKENS = 2048  # thinking model — leave room for thoughts plus the visible answer
_PROMPT = "What is 17 times 23? Reply with just the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "dev_chat.json"

# gemini-3-flash-preview published rates ($/token).
_RATES = {"input_rate": 0.50 / 1_000_000, "output_rate": 3.00 / 1_000_000, "cache_read_rate": 0.05 / 1_000_000}


@pytest.mark.verified
def test_chat_dev(
    scenario: Callable[..., Any],
    dev_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return dev_provider(auth, transport).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS)

    resp = scenario(_CASSETTE, _chat, requires="GEMINI_API_KEY")
    assert_chat(resp, provider="google")
    assert "391" in (resp.content or "")
    assert_cost(resp, **_RATES)
