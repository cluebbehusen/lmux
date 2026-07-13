"""AWS Bedrock reasoning (Converse thinking, Anthropic Claude): a step-by-step prompt
populates reasoning and states the answer, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 2048
_PROMPT = "What is 17 * 23? Think step by step, then state the number."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "reasoning.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return BedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, reasoning_effort="low"
    )


@pytest.mark.verified
def test_reasoning(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    assert_chat(resp, provider="aws-bedrock")
    assert resp.reasoning
    assert "391" in resp.content  # the thinking produced the correct answer (17 * 23)
    assert_cost(resp, **_RATES)
