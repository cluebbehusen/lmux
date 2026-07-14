"""Google embeddings on the Gemini Developer API (:batchEmbedContents): gemini-embedding-001
batches the whole list in one request (and reports no usageMetadata -> zero cost), while
gemini-embedding-2-preview routes to the same batch endpoint (NOT :embedContent, which is
Vertex-only) and does report tokens — the same-model backend divergence.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse

_G1_MODEL = "gemini-embedding-001"
_G2_MODEL = "gemini-embedding-2-preview"
_DIMENSIONS = 256  # outputDimensionality — also keeps recorded vectors small
_G1_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "dev_embed_batch.json"
_G2_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "dev_embed_g2.json"

_G1_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.0}
_G2_RATES = {"input_rate": 0.20 / 1_000_000, "output_rate": 0.0}


@pytest.mark.verified
def test_embeddings_batch(
    scenario: Callable[..., Any],
    dev_provider: Callable[..., Any],
    cosine_similarity: Callable[[list[float], list[float]], float],
    assert_cost: Callable[..., None],
) -> None:
    # One :batchEmbedContents request returns one embedding per input, in order. The Developer API
    # omits usageMetadata here, so input_tokens (and cost) fall back to zero.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return dev_provider(auth, transport).embed(_G1_MODEL, ["alpha", "beta", "gamma"], dimensions=_DIMENSIONS)

    resp = scenario(_G1_CASSETTE, _embed, requires="GEMINI_API_KEY")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 3
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert cosine_similarity(resp.embeddings[0], resp.embeddings[1]) < 0.999
    assert resp.usage is not None
    assert resp.usage.input_tokens == 0  # Dev-API batchEmbedContents carries no usageMetadata
    assert_cost(resp, **_G1_RATES)


@pytest.mark.verified
def test_embeddings_gemini2(
    scenario: Callable[..., Any],
    dev_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    # gemini-embedding-2 on the Dev API uses :batchEmbedContents (NOT the Vertex-only :embedContent)
    # and, unlike gemini-embedding-001 on this backend, does report token usage.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return dev_provider(auth, transport).embed(_G2_MODEL, ["alpha", "beta"], dimensions=_DIMENSIONS)

    resp = scenario(_G2_CASSETTE, _embed, requires="GEMINI_API_KEY")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 2
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert resp.usage is not None
    assert resp.usage.input_tokens > 0
    assert_cost(resp, **_G2_RATES)
