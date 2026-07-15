"""Azure AI Foundry embeddings: a deployment named ``embedding-3-small`` (serving
``text-embedding-3-small``) returns the requested vector, and cost keys off the response model at
the published rate.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse

_DEPLOYMENT = "embedding-3-small"  # deployment name
_MODEL = "text-embedding-3-small"  # model Azure reports back
_INPUT = "The quick brown fox jumps over the lazy dog."
_CASSETTE = Path(__file__).parent.parent / "cassettes" / "azure_foundry" / "embeddings.json"

# text-embedding-3-small published rate ($/token).
_RATES = {"input_rate": 0.02 / 1_000_000, "output_rate": 0.0}


@pytest.mark.verified
def test_embeddings(
    scenario: Callable[..., Any],
    foundry_provider: Callable[..., Any],
    assert_cost: Callable[..., None],
) -> None:
    def _embed(auth: Any, transport: Any) -> EmbeddingResponse:  # noqa: ANN401 — harness-supplied per mode
        return foundry_provider(auth, transport).embed(_DEPLOYMENT, _INPUT)

    resp = scenario(_CASSETTE, _embed, requires=("AZURE_FOUNDRY_KEY", "AZURE_FOUNDRY_ENDPOINT"))
    assert resp.provider == "azure-foundry"
    assert resp.model == _MODEL  # cost keys off the response model, not the deployment name
    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0]) == 1536
    assert resp.usage is not None
    assert resp.usage.output_tokens == 0
    assert_cost(resp, **_RATES)
