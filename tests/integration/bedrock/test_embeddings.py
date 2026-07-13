"""AWS Bedrock embeddings (InvokeModel, Amazon Titan v2): the requested dimensions are
returned, the same input embeds to near-identical vectors, and cost matches the rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse
from lmux_aws_bedrock.auth import BedrockSessionAuthProvider
from lmux_aws_bedrock.provider import BedrockProvider

_MODEL = "amazon.titan-embed-text-v2:0"
_REGION = "us-east-1"
_DIMENSIONS = 256
_INPUT = "The quick brown fox jumps over the lazy dog."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "bedrock" / "embeddings.json"

# amazon.titan-embed-text-v2 published rates ($/token, us-east-1).
_RATES = {"input_rate": 0.02 / 1_000_000, "output_rate": 0.0}


def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
    return BedrockProvider(auth=auth, transport=transport, region=_REGION).embed(_MODEL, _INPUT, dimensions=_DIMENSIONS)


@pytest.mark.verified
def test_embeddings(
    scenario: Callable[..., Any],
    assert_cost: Callable[..., None],
    cosine_similarity: Callable[[list[float], list[float]], float],
    offline_auth: BedrockSessionAuthProvider,
) -> None:
    first = scenario(_CASSETTE, _embed, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)
    second = scenario(_CASSETTE, _embed, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_auth)

    assert first.provider == "aws-bedrock"
    assert len(first.embeddings) == 1
    assert len(first.embeddings[0]) == _DIMENSIONS
    assert cosine_similarity(first.embeddings[0], second.embeddings[0]) == pytest.approx(1.0, abs=1e-3)
    assert first.usage is not None
    assert first.usage.output_tokens == 0
    assert_cost(first, **_RATES)
