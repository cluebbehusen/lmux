"""Integration + live-mode harness (moto-style: one contract, mock or real backend).

These tests live outside the coverage-gated unit suite (`testpaths` only covers
`packages/*/tests`), so run them explicitly and without coverage. Live/record
modes are gated by env vars (CI-friendly, and avoids a repo-root conftest):

    uv run pytest --no-cov tests/integration                                  # offline: replay cassettes
    OPENAI_API_KEY=... LMUX_LIVE=1   uv run pytest --no-cov tests/integration  # replay + live parity
    OPENAI_API_KEY=... LMUX_RECORD=1 uv run pytest --no-cov tests/integration -m record  # refresh cassettes

`--no-cov` MUST come before the path: the project's `addopts` has a bare `--cov`
whose optional argument would otherwise swallow the path as its coverage source.

Default runs replay recorded real responses (deterministic, no network/keys).
`LMUX_LIVE=1` additionally hits the real endpoint and asserts the same contract.
`LMUX_RECORD=1` (implies live) refreshes cassettes from real responses.

This module is the shared harness. A per-provider test is thin: point at cassette
files, declare the model's published rates, and call the `mount_cassette`,
`assert_chat`, `assert_cost`, `record_cassette`, and `cache_prompt` fixtures.
"""

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from lmux.types import ChatChunk, ChatResponse, EmbeddingResponse, ResponseResponse

_LIVE = os.environ.get("LMUX_LIVE") == "1"
_RECORD = os.environ.get("LMUX_RECORD") == "1"

_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls"}
_SECRET_KEYS = {"authorization", "api_key", "api-key", "openai-organization", "openai-project", "x-api-key"}


# MARK: Gating


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: full-stack test over the httpx transport (cassette-backed).")
    config.addinivalue_line("markers", "live: hits a real provider endpoint; skipped unless LMUX_LIVE=1.")
    config.addinivalue_line("markers", "record: re-records cassettes; skipped unless LMUX_RECORD=1.")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    live = _LIVE or _RECORD  # recording implies hitting the live endpoint
    for item in items:
        if "record" in item.keywords and not _RECORD:
            item.add_marker(pytest.mark.skip(reason="needs LMUX_RECORD=1"))
        elif "live" in item.keywords and not live:
            item.add_marker(pytest.mark.skip(reason="needs LMUX_LIVE=1"))


# MARK: Credentials


@pytest.fixture
def openai_key() -> str:
    """The real OpenAI key from the ambient environment; skip the test when absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_key() -> str:
    """The real Anthropic key from the ambient environment; skip the test when absent."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def groq_key() -> str:
    """The real Groq key from the ambient environment; skip the test when absent."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY not set")
    return key


@pytest.fixture
def gemini_key() -> str:
    """The real Gemini Developer API key from the environment; skip the test when absent."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    return key


# MARK: Prompt helpers


@pytest.fixture
def cache_prompt() -> Callable[[], str]:
    """A >1024-token prompt with a unique prefix, so call 1 is guaranteed a cold
    cache write and an identical call 2 is a warm cache read."""

    def _build(sentences: int = 180) -> str:
        nonce = uuid.uuid4().hex
        filler = " ".join(f"Sentence {i}: the quick brown fox jumps over the lazy dog." for i in range(sentences))
        return f"{nonce}\n\n{filler}\n\nReply with exactly the word: pong"

    return _build


# MARK: Contracts (shared across providers)


@pytest.fixture
def assert_chat() -> Callable[..., None]:
    """Structural contract shared by cassette and live chat responses."""

    def _assert(resp: ChatResponse, *, provider: str) -> None:
        assert isinstance(resp, ChatResponse)
        assert resp.provider == provider
        assert isinstance(resp.content, str)
        assert resp.usage is not None
        assert resp.finish_reason in _FINISH_REASONS

    return _assert


@pytest.fixture
def assert_cost() -> Callable[..., None]:
    """The core contract: ``resp.cost`` equals an independent recompute from
    ``resp.usage`` at the model's published rates. Holds for cassette usage and
    for whatever the live API returns. Rates are passed in per model (the
    independent source of truth — deliberately not read from the pricing table
    under test, so a bad edit there fails this).

    ``cache_write_rate_by_ttl`` recomputes cache-creation cost from
    ``usage.cache_creation_tokens_by_ttl`` (Anthropic/Bedrock 5m-vs-1h writes);
    it takes precedence over the flat ``cache_write_rate``. ``multiplier`` scales
    every expected cost (Vertex regional / Azure deployment-type / inference_geo
    uplifts, which the provider applies linearly to the base cost)."""

    def _assert(  # noqa: PLR0913
        resp: ChatResponse | ChatChunk | EmbeddingResponse | ResponseResponse,
        *,
        input_rate: float,
        output_rate: float,
        cache_read_rate: float = 0.0,
        cache_write_rate: float = 0.0,
        cache_write_rate_by_ttl: dict[str, float] | None = None,
        multiplier: float = 1.0,
    ) -> None:
        usage = resp.usage
        assert usage is not None
        read = usage.cache_read_tokens or 0
        write = usage.cache_creation_tokens or 0
        billable_input = usage.input_tokens - read - write

        exp_input = billable_input * input_rate * multiplier
        exp_output = usage.output_tokens * output_rate * multiplier
        exp_read = read * cache_read_rate * multiplier if read else None
        if cache_write_rate_by_ttl is not None:
            by_ttl = usage.cache_creation_tokens_by_ttl or {}
            exp_write = (
                sum(toks * cache_write_rate_by_ttl[ttl] for ttl, toks in by_ttl.items()) * multiplier
                if by_ttl
                else None
            )
        else:
            exp_write = write * cache_write_rate * multiplier if write else None
        exp_total = exp_input + exp_output + (exp_read or 0.0) + (exp_write or 0.0)

        assert resp.cost is not None
        assert resp.cost.input_cost == pytest.approx(exp_input)
        assert resp.cost.output_cost == pytest.approx(exp_output)
        assert resp.cost.cache_read_cost == (pytest.approx(exp_read) if exp_read is not None else None)
        assert resp.cost.cache_creation_cost == (pytest.approx(exp_write) if exp_write is not None else None)
        assert resp.cost.total_cost == pytest.approx(exp_total)

    return _assert


# MARK: Cassette replay / record


@pytest.fixture
def mount_cassette(respx_mock: respx.MockRouter) -> Callable[[Path], dict[str, Any]]:
    """Mount a recorded JSON cassette on respx so the provider's real transport
    replays it. Returns the parsed cassette for further assertions."""

    def _mount(cassette_path: Path) -> dict[str, Any]:
        cassette = json.loads(cassette_path.read_text())
        if "response_sse" in cassette:  # streaming cassette: raw SSE bytes
            response = httpx.Response(
                200, content=cassette["response_sse"].encode(), headers={"content-type": "text/event-stream"}
            )
        else:  # unary JSON cassette
            response = httpx.Response(200, json=cassette["response"])
        respx_mock.post(cassette["endpoint"]).mock(return_value=response)
        return cassette

    return _mount


# MARK: Request capture + hand-authored mounting


@pytest.fixture
def sent_request(respx_mock: respx.MockRouter) -> Callable[..., tuple[dict[str, Any], dict[str, str]]]:
    """The JSON body + headers of an outgoing request (default: the most recent).

    The core harness mocks only responses; this exposes the REQUEST the provider
    emitted, so tests can assert request-shaping (cache breakpoints, endpoint
    routing, dropped params) and auth headers. Pass a positional index for a
    specific call in a fan-out (0 = first)."""

    def _get(index: int = -1) -> tuple[dict[str, Any], dict[str, str]]:
        request = respx_mock.calls[index].request
        body = json.loads(request.content) if request.content else {}
        return body, dict(request.headers)

    return _get


@pytest.fixture
def mount_response(respx_mock: respx.MockRouter) -> Callable[..., None]:
    """Mount one response of any status / body / headers on an endpoint — for
    replaying recorded non-2xx / error responses that ``mount_cassette`` (which
    always replays HTTP 200) can't express."""

    def _mount(  # noqa: PLR0913
        endpoint: str,
        *,
        status: int = 200,
        json_body: Any = None,  # noqa: ANN401
        text: str | None = None,
        sse: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        extra = headers or {}
        if sse is not None:
            response = httpx.Response(
                status, content=sse.encode(), headers={"content-type": "text/event-stream", **extra}
            )
        elif text is not None:
            response = httpx.Response(status, content=text.encode(), headers=extra)
        else:
            response = httpx.Response(status, json=json_body, headers=extra)
        respx_mock.post(endpoint).mock(return_value=response)

    return _mount


@pytest.fixture
def mount_sequence(respx_mock: respx.MockRouter) -> Callable[..., None]:
    """Mount N JSON responses served in order on a single endpoint, modelling
    provider fan-out (Vertex ``:predict`` batch-size-1, ``:embedContent``, Bedrock
    multi-input) where one call issues N POSTs to the same URL. Assert the fan-out
    with ``respx_mock.calls.call_count == N``."""

    def _mount(endpoint: str, responses: list[dict[str, Any]]) -> None:
        side_effect = [httpx.Response(r.get("status", 200), json=r["json"]) for r in responses]
        respx_mock.post(endpoint).mock(side_effect=side_effect)

    return _mount


def _scrub(obj: object) -> object:
    """Defensive: drop any credential-ish keys before a cassette touches disk."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if str(k).lower() not in _SECRET_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


@pytest.fixture
def record_cassette() -> Callable[..., dict[str, Any]]:
    """Hit a real endpoint, scrub credentials, and write a JSON cassette. Returns
    the raw response JSON so the recorder can assert the captured shape."""

    def _record(
        cassette_path: Path, *, endpoint: str, request_body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        response = httpx.post(endpoint, headers=headers, json=request_body, timeout=90.0)
        response.raise_for_status()
        data = response.json()
        cassette = {"endpoint": endpoint, "request": _scrub(request_body), "response": _scrub(data)}
        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        cassette_path.write_text(json.dumps(cassette, indent=2) + "\n")
        return data

    return _record


@pytest.fixture
def record_stream_cassette() -> Callable[..., str]:
    """Like ``record_cassette`` but for a streaming (SSE) endpoint: captures the
    raw event-stream body verbatim and returns it for assertions."""

    def _record(cassette_path: Path, *, endpoint: str, request_body: dict[str, Any], headers: dict[str, str]) -> str:
        response = httpx.post(endpoint, headers=headers, json=request_body, timeout=90.0)
        response.raise_for_status()
        sse = response.text
        cassette = {"endpoint": endpoint, "request": _scrub(request_body), "response_sse": sse}
        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        cassette_path.write_text(json.dumps(cassette, indent=2) + "\n")
        return sse

    return _record
