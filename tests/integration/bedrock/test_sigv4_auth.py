"""AWS Bedrock SigV4 auth: a request signed with resolved AWS credentials (not a bearer
token) is accepted by the Bedrock API, validating lmux's SigV4 signing end-to-end.

Live/record resolve real credentials (AWS_PROFILE) and sign; offline replays the cassette
after signing locally with the dummy-credential session.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "amazon.nova-micro-v1:0"
_REGION = "us-east-1"
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "sigv4_auth.json"

# amazon.nova-micro-v1 published rates ($/token, us-east-1).
_RATES = {"input_rate": 0.035 / 1_000_000, "output_rate": 0.14 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return BedrockProvider(
        auth=auth, transport=transport, region=_REGION, default_headers={"X-Lmux-Integration": "default-headers"}
    ).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=16)


@pytest.mark.verified
def test_sigv4_auth(
    scenario: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    # Force SigV4: the provider reads the bearer token env first, so remove it — live/record
    # then resolve real AWS credentials (AWS_PROFILE) and sign the request instead.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    resp = scenario(_CASSETTE, _chat, requires="AWS_PROFILE", offline_auth=offline_auth)
    assert_chat(resp, provider="aws-bedrock")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
