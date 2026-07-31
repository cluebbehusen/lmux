"""OpenAI Responses caches only the prefix before an explicit breakpoint."""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import CachePointContent, ResponseInputMessage, ResponseResponse, TextContent
from lmux_openai.params import OpenAIParams
from lmux_openai.provider import OpenAIProvider

_MODEL = "gpt-5.6-luna"
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "openai"
_WRITE_CASSETTE = _CASSETTES / "responses_explicit_cache_write.json"
_READ_CASSETTE = _CASSETTES / "responses_explicit_cache_read.json"


@pytest.mark.verified
def test_responses_explicit_cache(scenario: Callable[..., Any], cache_prompt: Callable[..., str]) -> None:
    stable = ResponseInputMessage(
        role="developer",
        content=[TextContent(text=cache_prompt()), CachePointContent()],
    )
    cache_key = uuid.uuid4().hex

    def _write(auth: Any, transport: Any) -> ResponseResponse:  # noqa: ANN401
        input_items = [stable, ResponseInputMessage(role="user", content="Request variant A.")]
        params = OpenAIParams(prompt_cache_key=cache_key)
        return OpenAIProvider(auth=auth, transport=transport).create_response(
            _MODEL, input_items, provider_params=params
        )

    write = scenario(_WRITE_CASSETTE, _write, requires="OPENAI_API_KEY")
    assert write.provider == "openai"
    assert "pong" in write.output_text.lower()
    assert write.usage is not None
    created = write.usage.cache_creation_tokens
    assert created is not None
    assert created > 1024
    assert write.usage.input_tokens > created
    assert write.usage.cache_read_tokens is None

    def _read(auth: Any, transport: Any) -> ResponseResponse:  # noqa: ANN401
        input_items = [stable, ResponseInputMessage(role="user", content="Request variant B.")]
        params = OpenAIParams(prompt_cache_key=cache_key)
        return OpenAIProvider(auth=auth, transport=transport).create_response(
            _MODEL, input_items, provider_params=params
        )

    read = scenario(_READ_CASSETTE, _read, requires="OPENAI_API_KEY")
    assert read.provider == "openai"
    assert "pong" in read.output_text.lower()
    assert read.usage is not None
    assert read.usage.cache_creation_tokens is None
    assert read.usage.cache_read_tokens == created
    assert read.usage.input_tokens > created
