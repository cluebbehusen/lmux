"""OpenAI Responses API cost parity — a recorded cassette and the live endpoint
satisfy the same cost contract over the /responses surface (distinct from
/chat/completions). ``gpt-4o-mini`` bills input+output only; a short prompt keeps
the recording cheap. assert_chat doesn't apply (a ResponseResponse has no
finish_reason), so this asserts output_text + cost.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ResponseResponse
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/responses"
_MODEL = "gpt-4o-mini"
_PROMPT = "Reply with exactly the word: pong"

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "responses.json"

# gpt-4o-mini published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _respond(auth: _FakeAuth | None) -> ResponseResponse:
    return OpenAIProvider(auth=auth).create_response(_MODEL, _PROMPT)


class TestResponsesCassette:
    @pytest.mark.integration
    def test_response_cost(
        self, mount_cassette: Callable[[Path], dict[str, Any]], assert_cost: Callable[..., None]
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _respond(_FakeAuth())
        assert resp.provider == "openai"
        assert isinstance(resp.output_text, str)
        assert resp.id
        assert_cost(resp, **_RATES)


class TestLiveResponses:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_response(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _respond(None)  # real OPENAI_API_KEY from env
        assert resp.provider == "openai"
        assert isinstance(resp.output_text, str)
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {"model": _MODEL, "input": _PROMPT}
        headers = {"Authorization": f"Bearer {openai_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert data["usage"]["input_tokens"] > 0, "expected input token usage"
