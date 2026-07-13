"""AWS Bedrock chat (Converse, Amazon Nova): a deterministic prompt returns the
expected word, and cost matches the published rate.
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
_MAX_TOKENS = 16
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "chat.json"

# amazon.nova-micro-v1 published rates ($/token, us-east-1).
_RATES = {"input_rate": 0.035 / 1_000_000, "output_rate": 0.14 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return BedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS
    )


@pytest.mark.verified
def test_chat(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    assert_chat(resp, provider="aws-bedrock")
    assert "pong" in (resp.content or "").lower()
    assert_cost(resp, **_RATES)
