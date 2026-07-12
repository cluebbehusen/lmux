"""OpenAI embeddings cost parity — input-only cost over a unary JSON response.

``text-embedding-3-small`` bills input tokens only (output rate 0), so this
exercises the simplest cost path and confirms ``assert_cost`` handles a non-chat
response. ``dimensions=16`` keeps the recorded vector (and cassette) tiny.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import EmbeddingResponse
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/embeddings"
_MODEL = "text-embedding-3-small"
_DIMENSIONS = 16
_INPUT = "The quick brown fox jumps over the lazy dog."

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "embeddings.json"

# text-embedding-3-small published rates ($ per token) — independent source of truth.
_RATES = {"input_rate": 0.02 / 1_000_000, "output_rate": 0.0}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _embed(auth: _FakeAuth | None) -> EmbeddingResponse:
    return OpenAIProvider(auth=auth).embed(_MODEL, _INPUT, dimensions=_DIMENSIONS)


class TestEmbeddingCassette:
    @pytest.mark.integration
    def test_embedding_cost(
        self, mount_cassette: Callable[[Path], dict[str, Any]], assert_cost: Callable[..., None]
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _embed(_FakeAuth())
        assert resp.provider == "openai"
        assert resp.embeddings
        assert resp.usage is not None
        assert resp.usage.output_tokens == 0
        assert_cost(resp, **_RATES)


class TestLiveEmbedding:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_embedding(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _embed(None)  # real OPENAI_API_KEY from env
        assert resp.provider == "openai"
        assert resp.embeddings
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {"model": _MODEL, "input": _INPUT, "dimensions": _DIMENSIONS}
        headers = {"Authorization": f"Bearer {openai_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert data["usage"]["prompt_tokens"] > 0, "expected input token usage"
