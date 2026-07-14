"""Google streaming on Vertex AI: the streamed content is the expected word, and the
terminal usage-bearing chunk's cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_stream.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


@pytest.mark.verified
def test_streaming(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401 — harness-supplied per mode
        return list(
            vertex_provider(auth, transport).chat_stream(
                _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
            )
        )

    chunks = scenario(_CASSETTE, _stream, requires="VERTEXAI_API_KEY")
    content = "".join(c.delta or "" for c in chunks)
    assert "pong" in content.lower()
    assert any(c.finish_reason for c in chunks)
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost
    assert_cost(with_cost[-1], **_RATES)
