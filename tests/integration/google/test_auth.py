"""Google ADC auth smoke test: a Vertex chat authenticated with Application Default Credentials
(OAuth bearer + x-goog-user-project quota project) — the production auth path for enterprise Vertex
that API keys do not exercise. Offline uses a fake-Credentials stub so the bearer/quota headers are
built (but never sent); live/record use real ADC against Vertex.
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
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_adc_auth.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


@pytest.mark.verified
def test_vertex_adc_auth(
    scenario: Callable[..., Any],
    vertex_adc_provider: Callable[..., Any],
    offline_adc_auth: Any,  # noqa: ANN401 — fake-Credentials stub for offline
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_adc_provider(auth, transport).chat(
            _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
        )

    resp = scenario(_CASSETTE, _chat, requires="GOOGLE_CLOUD_PROJECT", offline_auth=offline_adc_auth)
    assert_chat(resp, provider="google")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
