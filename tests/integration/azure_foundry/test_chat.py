"""Azure AI Foundry chat: a deployment named ``Phi-4-mini`` (serving the model
``Phi-4-mini-reasoning``) returns the expected answer, and cost keys off the model reported in the
response, NOT the deployment name, at the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage

_DEPLOYMENT = "Phi-4-mini"  # deployment name (goes in the URL path)
_MODEL = "Phi-4-mini-reasoning"  # model Azure reports back (what cost keys off)
_MAX_TOKENS = 2048  # reasoning model — leave room for the inline chain-of-thought plus the answer
_PROMPT = "What is 17 times 23? Reply with just the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "azure_foundry" / "chat.json"

# Phi-4-mini-reasoning published rates ($/token).
_RATES = {"input_rate": 0.075 / 1_000_000, "output_rate": 0.30 / 1_000_000}


@pytest.mark.verified
def test_chat(
    scenario: Callable[..., Any],
    foundry_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return foundry_provider(auth, transport).chat(
            _DEPLOYMENT, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
        )

    resp = scenario(_CASSETTE, _chat, requires=("AZURE_FOUNDRY_KEY", "AZURE_FOUNDRY_ENDPOINT"))
    assert_chat(resp, provider="azure-foundry")
    # The deployment is "Phi-4-mini" but the response model is "Phi-4-mini-reasoning"; cost keys off
    # the latter. This proves the provider prices from the response, not the deployment string.
    assert resp.model == _MODEL
    assert "391" in (resp.content or "")
    assert_cost(resp, **_RATES)
