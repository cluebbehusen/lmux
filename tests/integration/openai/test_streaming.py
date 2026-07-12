"""OpenAI streaming: the streamed content includes the expected word, and the
terminal usage-bearing chunk's cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 32
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "chat_stream.json"

# gpt-4o-mini published rates ($/token).
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401 — harness-supplied per mode
    provider = OpenAIProvider(auth=auth, transport=transport)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


@pytest.mark.verified
def test_streaming(scenario: Callable[..., Any], assert_cost: Callable[..., None]) -> None:
    chunks = scenario(_CASSETTE, _stream, requires="OPENAI_API_KEY")
    content = "".join(c.delta or "" for c in chunks)
    assert "pong" in content.lower()
    assert any(c.finish_reason for c in chunks)
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost
    assert_cost(with_cost[-1], **_RATES)
