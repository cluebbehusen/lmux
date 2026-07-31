"""Native Anthropic-on-Bedrock SigV4 auth: a request signed with resolved AWS credentials
(not a bearer token) is accepted by the Bedrock API, validating the shared SigV4 signing
end-to-end on the InvokeModel path.

Live/record resolve real credentials (AWS_PROFILE, e.g. an SSO session) and sign; offline
replays the cassette after signing locally with the dummy-credential session.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "bedrock_sigv4_auth.json"
_DEFAULT_HEADERS = {
    "X-Lmux-Integration": "replaced",
    "x-lmux-integration": "default  headers",
    "User-Agent": "lmux-integration",
}

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicBedrockProvider(
        auth=auth, transport=transport, region=_REGION, default_headers=_DEFAULT_HEADERS
    ).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS)


@pytest.mark.verified
def test_sigv4_auth_bedrock(
    scenario: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    # Force SigV4: the provider reads the bearer token env first, so remove it — live/record
    # then resolve real AWS credentials (AWS_PROFILE) and sign the request instead.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    resp = scenario(_CASSETTE, _chat, requires="AWS_PROFILE", offline_auth=offline_bedrock_auth)
    assert_chat(resp, provider="anthropic-bedrock")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
