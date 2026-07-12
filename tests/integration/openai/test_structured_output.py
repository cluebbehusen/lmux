"""OpenAI structured output (json_schema) — recorded + live.

``response_format=JsonSchemaResponseFormat`` makes the model return ``content``
that is a JSON string conforming to the schema; assert the parsed structure and
cost. ``gpt-4o-mini`` keeps it cheap and 17*23 is deterministic.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, JsonSchemaResponseFormat, UserMessage
from lmux_openai.provider import OpenAIProvider

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 64
_PROMPT = "What is 17 times 23?"
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT = JsonSchemaResponseFormat(name="math_answer", json_schema=_SCHEMA, strict=True)

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "openai" / "structured_output.json"

# gpt-4o-mini published rates ($/token) — the independent source of truth.
_RATES = {"input_rate": 0.15 / 1_000_000, "output_rate": 0.60 / 1_000_000}


class _FakeAuth:
    """Mock-mode auth: respx intercepts the request, so the key is never used."""

    def get_credentials(self) -> str:
        return "sk-mock-not-used"

    async def aget_credentials(self) -> str:
        return "sk-mock-not-used"


def _chat(auth: _FakeAuth | None) -> ChatResponse:
    return OpenAIProvider(auth=auth).chat(
        _MODEL, [UserMessage(content=_PROMPT)], response_format=_RESPONSE_FORMAT, max_tokens=_MAX_TOKENS
    )


class TestStructuredCassette:
    @pytest.mark.integration
    def test_structured_output_and_cost(
        self,
        mount_cassette: Callable[[Path], dict[str, Any]],
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        mount_cassette(_CASSETTE)
        resp = _chat(_FakeAuth())
        assert_chat(resp, provider="openai")
        assert resp.content is not None
        assert json.loads(resp.content)["answer"] == 391
        assert_cost(resp, **_RATES)


class TestLiveStructured:
    @pytest.mark.integration
    @pytest.mark.live
    def test_live_structured(
        self,
        openai_key: str,  # noqa: ARG002 — requested to skip when unset
        assert_chat: Callable[..., None],
        assert_cost: Callable[..., None],
    ) -> None:
        resp = _chat(None)  # real OPENAI_API_KEY from env
        assert_chat(resp, provider="openai")
        assert resp.content is not None
        assert json.loads(resp.content)["answer"] == 391
        assert_cost(resp, **_RATES)


class TestRecordCassette:
    @pytest.mark.live
    @pytest.mark.record
    def test_record(self, openai_key: str, record_cassette: Callable[..., dict[str, Any]]) -> None:
        body = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _PROMPT}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "math_answer", "strict": True, "schema": _SCHEMA},
            },
            "max_tokens": _MAX_TOKENS,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {openai_key}"}
        data = record_cassette(_CASSETTE, endpoint=_ENDPOINT, request_body=body, headers=headers)
        assert json.loads(data["choices"][0]["message"]["content"])["answer"] == 391, "expected structured answer"
