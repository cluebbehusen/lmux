"""Azure AI Foundry Entra ID auth smoke test: a chat authenticated with an Azure AD bearer token
(``DefaultAzureCredential``) instead of an API key, exercising the ``Authorization: Bearer`` path.
Offline uses a fake static token to build the header (never sent); live/record use az login /
``DefaultAzureCredential`` against the real resource.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage

_DEPLOYMENT = "Phi-4-mini"
_MODEL = "Phi-4-mini-reasoning"
_MAX_TOKENS = 256
_PROMPT = "Reply with the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "azure_foundry" / "ad_auth.json"

_RATES = {"input_rate": 0.075 / 1_000_000, "output_rate": 0.30 / 1_000_000}


@pytest.mark.verified
def test_azure_ad_auth(
    scenario: Callable[..., Any],
    foundry_ad_provider: Callable[..., Any],
    offline_ad_auth: Any,  # noqa: ANN401 — fake-token stub for offline
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return foundry_ad_provider(auth, transport).chat(
            _DEPLOYMENT, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
        )

    resp = scenario(_CASSETTE, _chat, requires="AZURE_FOUNDRY_ENDPOINT", offline_auth=offline_ad_auth)
    assert_chat(resp, provider="azure-foundry")
    assert resp.model == _MODEL
    assert_cost(resp, **_RATES)
