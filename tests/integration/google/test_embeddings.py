"""Google embeddings on Vertex AI (:predict): two models exercise the two Vertex batching
behaviors — gemini-embedding-001 fans out to one request per input, while text-embedding-005
batches the whole list in one request. Both report tokens via statistics.token_count.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse

_FANOUT_MODEL = "gemini-embedding-001"
_BATCH_MODEL = "text-embedding-005"
_DIMENSIONS = 256  # outputDimensionality — also keeps recorded vectors small
_FANOUT_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_embed_fanout.json"
_BATCH_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_embed_batch.json"

_FANOUT_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.0}
_BATCH_RATES = {"input_rate": 0.10 / 1_000_000, "output_rate": 0.0}


@pytest.mark.verified
def test_embeddings_predict_fanout(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    # gemini-embedding-001 accepts one input per :predict request, so a 3-item list is issued as
    # three requests and the tokens are summed. Offline replays the single-embedding cassette per
    # request, so the count (3) and summed usage — not vector identity — are what this proves.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).embed(_FANOUT_MODEL, ["alpha", "beta", "gamma"], dimensions=_DIMENSIONS)

    resp = scenario(_FANOUT_CASSETTE, _embed, requires="VERTEXAI_API_KEY")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 3
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert resp.usage.input_tokens > 0
    assert_cost(resp, **_FANOUT_RATES)


@pytest.mark.verified
def test_embeddings_predict_batch(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    cosine_similarity: Callable[[list[float], list[float]], float],
    assert_cost: Callable[..., None],
) -> None:
    # text-embedding-005 batches the whole list in one :predict request, so both embeddings are in
    # the single cassette response and come back distinct and in order.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).embed(_BATCH_MODEL, ["alpha", "beta"], dimensions=_DIMENSIONS)

    resp = scenario(_BATCH_CASSETTE, _embed, requires="VERTEXAI_API_KEY")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 2
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert cosine_similarity(resp.embeddings[0], resp.embeddings[1]) < 0.999  # distinct inputs -> distinct vectors
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert_cost(resp, **_BATCH_RATES)
