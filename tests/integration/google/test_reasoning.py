"""Google reasoning on Vertex AI: reasoning_effort maps to thinkingConfig, the response
carries thinking tokens, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 2048  # leave room for the thinking budget (effort=low -> 1024) plus the answer
_PROMPT = "What is 17 times 23? Reply with just the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_reasoning.json"
_CASSETTE_DIR = _CASSETTE.parent
_MODEL_CASES = [
    ("gemini-2.5-flash", _CASSETTE_DIR / "vertex_reasoning_2_5_flash_high.json"),
    ("gemini-2.5-flash-lite", _CASSETTE_DIR / "vertex_reasoning_2_5_flash_lite_high.json"),
    ("gemini-2.5-pro", _CASSETTE_DIR / "vertex_reasoning_2_5_pro_high.json"),
    ("gemini-3.5-flash", _CASSETTE_DIR / "vertex_reasoning_3_5_flash_high.json"),
]

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}


@pytest.mark.verified
def test_reasoning(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(
            _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, reasoning_effort="low"
        )

    resp = scenario(_CASSETTE, _chat, requires=("VERTEXAI_API_KEY", "GOOGLE_CLOUD_PROJECT"))
    assert_chat(resp, provider="google")
    assert "391" in (resp.content or "")
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens is not None
    assert resp.usage.reasoning_tokens > 0
    # output_tokens folds in the thinking tokens (Gemini bills them at the output rate), so it exceeds
    # the reasoning sub-count; before the fix output_tokens was just the few visible tokens. This makes
    # assert_cost bill the thinking tokens. (Live-safe: pins the invariant, not the exact thought count.)
    assert resp.usage.output_tokens > resp.usage.reasoning_tokens
    assert_cost(resp, **_RATES)


@pytest.mark.verified
@pytest.mark.parametrize(("model", "cassette"), _MODEL_CASES)
def test_reasoning_effort_model_matrix(
    model: str,
    cassette: Path,
    scenario: Callable[..., Any],
    vertex_adc_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_adc_provider(auth, transport).chat(
            model,
            [UserMessage(content=_PROMPT)],
            reasoning_effort="high",
        )

    resp = scenario(cassette, _chat, requires="GOOGLE_CLOUD_PROJECT")
    assert_chat(resp, provider="google")
    assert "391" in (resp.content or "")
    assert resp.reasoning
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens is not None
    assert resp.usage.reasoning_tokens > 0
    assert resp.usage.output_tokens > resp.usage.reasoning_tokens
