"""Google embeddings on Vertex AI (:predict): both models batch the caller's list in one
native request and report tokens via statistics.token_count.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.exceptions import InvalidRequestError
from lmux.types import EmbeddingResponse

_GEMINI_MODEL = "gemini-embedding-001"
_TEXT_MODEL = "text-embedding-005"
_DIMENSIONS = 256  # outputDimensionality — also keeps recorded vectors small
_GEMINI_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_embed_gemini_batch.json"
_TEXT_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_embed_batch.json"

_GEMINI_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.0}
_TEXT_RATES = {"input_rate": 0.10 / 1_000_000, "output_rate": 0.0}


@pytest.mark.verified
def test_embeddings_predict_gemini_batch(
    scenario: Callable[..., Any],
    vertex_adc_provider: Callable[..., Any],
    cosine_similarity: Callable[[list[float], list[float]], float],
    assert_cost: Callable[..., None],
) -> None:
    # The native response contains all three embeddings, proving the caller's list survives the
    # batch boundary. Exact request shape and request count are pinned by provider unit tests.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_adc_provider(auth, transport).embed(
            _GEMINI_MODEL, ["alpha", "beta", "gamma"], dimensions=_DIMENSIONS
        )

    resp = scenario(_GEMINI_CASSETTE, _embed, requires="GOOGLE_CLOUD_PROJECT")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 3
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert cosine_similarity(resp.embeddings[0], resp.embeddings[1]) < 0.999
    assert cosine_similarity(resp.embeddings[1], resp.embeddings[2]) < 0.999
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert resp.usage.input_tokens == 3
    assert_cost(resp, **_GEMINI_RATES)


@pytest.mark.live
def test_embeddings_predict_gemini_native_batch_limit(
    scenario: Callable[..., Any], vertex_adc_provider: Callable[..., Any]
) -> None:
    texts = [f"native batch limit input {index}" for index in range(251)]

    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_adc_provider(auth, transport).embed(_GEMINI_MODEL, texts, dimensions=_DIMENSIONS)

    with pytest.raises(InvalidRequestError, match="too many instances"):
        scenario(_GEMINI_CASSETTE, _embed, requires="GOOGLE_CLOUD_PROJECT")


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
        return vertex_provider(auth, transport).embed(_TEXT_MODEL, ["alpha", "beta"], dimensions=_DIMENSIONS)

    resp = scenario(_TEXT_CASSETTE, _embed, requires=("VERTEXAI_API_KEY", "GOOGLE_CLOUD_PROJECT"))
    assert resp.provider == "google"
    assert len(resp.embeddings) == 2
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert cosine_similarity(resp.embeddings[0], resp.embeddings[1]) < 0.999  # distinct inputs -> distinct vectors
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert_cost(resp, **_TEXT_RATES)
