"""AWS Bedrock streaming (ConverseStream, Amazon Nova): the streamed content includes
the expected word, and the terminal usage-bearing chunk's cost matches the rate.

Bedrock streams AWS binary event-stream frames (not SSE), recorded verbatim as raw bytes.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatChunk, UserMessage
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "amazon.nova-micro-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 32
_PROMPT = "Reply with exactly the word: pong"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "chat_stream.json"

# amazon.nova-micro-v1 published rates ($/token, us-east-1).
_RATES = {"input_rate": 0.035 / 1_000_000, "output_rate": 0.14 / 1_000_000}


def _stream(auth: Any, transport: Any) -> list[ChatChunk]:  # noqa: ANN401 — harness-supplied per mode
    provider = BedrockProvider(auth=auth, transport=transport, region=_REGION)
    return list(provider.chat_stream(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS))


@pytest.mark.verified
def test_streaming(
    scenario: Callable[..., Any],
    assert_cost: Callable[..., None],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    chunks = scenario(_CASSETTE, _stream, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    content = "".join(c.delta or "" for c in chunks)
    assert "pong" in content.lower()
    assert any(c.finish_reason for c in chunks)
    with_cost = [c for c in chunks if c.cost is not None]
    assert with_cost
    assert_cost(with_cost[-1], **_RATES)
