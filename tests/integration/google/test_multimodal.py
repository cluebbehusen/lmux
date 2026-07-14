"""Google multimodal input on Vertex AI: an inline base64 image (data URI) is sent as
inlineData and the model describes it, exercising the image-content wire mapping.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, ImageContent, TextContent, UserMessage
from lmux_google.params import GoogleParams

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 16
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_multimodal.json"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000}

# A 48x48 solid crimson PNG as a data URI (maps to inlineData).
_RED_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAAaElEQVR4nM3OQQEAIBCAMCSDceyfwjD+r4Asw"
    "dbdhxKJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkRiJkR"
    "iJkRiJkRiJkRh/B6YHEOMBjO65d3oAAAAASUVORK5CYII="
)


@pytest.mark.verified
def test_multimodal(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        content = [
            TextContent(text="What color is this square? Answer with one word."),
            ImageContent(url=_RED_PNG),
        ]
        return vertex_provider(auth, transport).chat(
            _MODEL, [UserMessage(content=content)], max_tokens=_MAX_TOKENS, provider_params=_NO_THINK
        )

    resp = scenario(_CASSETTE, _chat, requires=("VERTEXAI_API_KEY", "GOOGLE_CLOUD_PROJECT"))
    assert_chat(resp, provider="google")
    assert "red" in (resp.content or "").lower() or "crimson" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
