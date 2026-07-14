"""Google async chat on Vertex AI (offline replay): achat drives the async client and the
async header path, reusing the sync chat cassette. The recording transport is sync-only, so
this scenario is offline-only.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_chat.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


@pytest.mark.offline
def test_achat(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _achat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        async def _run() -> ChatResponse:
            return await vertex_provider(auth, transport, async_=True).achat(
                _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
            )

        return asyncio.run(_run())

    resp = scenario(_CASSETTE, _achat, requires="VERTEXAI_API_KEY")
    assert_chat(resp, provider="google")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
