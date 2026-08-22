"""Google embeddings on Vertex :embedContent (gemini-embedding-2, ADC): the third Vertex
embeddings wire path — single content per request, fanned out one request per input. This
endpoint rejects API keys, so it authenticates with ADC (OAuth bearer), and gemini-embedding-2
is served only regionally (us-central1), not on the global endpoint.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse
from lmux_google import GoogleParams

_MODEL = "gemini-embedding-2-preview"
_LOCATION = "us-central1"
_DIMENSIONS = 256
_TASK_TYPE = "RETRIEVAL_DOCUMENT"
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "google" / "vertex_embed_content.json"

_RATES = {"input_rate": 0.20 / 1_000_000, "output_rate": 0.0}


@pytest.mark.verified
def test_embeddings_embed_content(
    scenario: Callable[..., Any],
    vertex_adc_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    # gemini-embedding-2 routes to :embedContent (single content/request) on Vertex, so a 2-item list
    # fans out to two requests and the tokens are summed. Offline replays the single-content cassette
    # per request, so the count (2) and summed usage — not vector identity — are what this proves.
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_adc_provider(auth, transport, location=_LOCATION).embed(
            _MODEL,
            ["alpha", "beta"],
            dimensions=_DIMENSIONS,
            provider_params=GoogleParams(task_type=_TASK_TYPE),
        )

    resp = scenario(_CASSETTE, _embed, requires="GOOGLE_CLOUD_PROJECT")
    assert resp.provider == "google"
    assert len(resp.embeddings) == 2
    assert all(len(v) == _DIMENSIONS for v in resp.embeddings)
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert resp.usage.input_tokens > 0
    assert_cost(resp, **_RATES)
