"""Azure AI Foundry streaming: the streamed content contains the expected word, and the terminal
usage-bearing chunk's cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage

_DEPLOYMENT = "Phi-4-mini"
_MAX_TOKENS = 512
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "azure_foundry" / "stream.json"

_RATES = {"input_rate": 0.075 / 1_000_000, "output_rate": 0.30 / 1_000_000}


@pytest.mark.verified
def test_streaming(
    scenario: Callable[..., Any],
    foundry_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401 — harness-supplied per mode
        return list(
            foundry_provider(auth, transport).chat_stream(
                _DEPLOYMENT, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
            )
        )

    chunks = scenario(_CASSETTE, _stream, requires=("AZURE_FOUNDRY_KEY", "AZURE_FOUNDRY_ENDPOINT"))
    content = "".join(c.delta or "" for c in chunks)
    assert "pong" in content.lower()
    assert any(c.finish_reason for c in chunks)
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost
    assert_cost(with_cost[-1], **_RATES)
