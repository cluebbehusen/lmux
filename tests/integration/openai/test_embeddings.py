"""OpenAI embeddings: the requested dimensions are returned, the same input embeds
to near-identical vectors, and cost matches the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse
from lmux_openai.provider import OpenAIProvider

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 16
_INPUT = "The quick brown fox jumps over the lazy dog."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "embeddings.json"

# text-embedding-3-small published rates ($/token).
_RATES = {"input_rate": 0.02 / 1_000_000, "output_rate": 0.0}


def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
    return OpenAIProvider(auth=auth, transport=transport).embed(_MODEL, _INPUT, dimensions=_DIMENSIONS)


@pytest.mark.verified
def test_embeddings(
    scenario: Callable[..., Any],
    assert_cost: Callable[..., None],
    cosine_similarity: Callable[[list[float], list[float]], float],
) -> None:
    first = scenario(_CASSETTE, _embed, requires="OPENAI_API_KEY")
    second = scenario(_CASSETTE, _embed, requires="OPENAI_API_KEY")

    assert first.provider == "openai"
    assert len(first.embeddings) == 1
    assert len(first.embeddings[0]) == _DIMENSIONS
    assert cosine_similarity(first.embeddings[0], second.embeddings[0]) == pytest.approx(1.0, abs=1e-3)
    assert first.usage is not None
    assert first.usage.output_tokens == 0
    assert_cost(first, **_RATES)
