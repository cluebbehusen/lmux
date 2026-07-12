"""Integration harness: each test runs offline (replay a cassette), live (real API),
or record (refresh the cassette from a real call), chosen by env var.

These tests live outside the coverage-gated unit suite (`testpaths` covers only
`packages/*/tests`), so run them explicitly and without coverage:

    uv run pytest --no-cov tests/integration                # offline (default)
    LMUX_LIVE=1   uv run pytest --no-cov tests/integration  # live
    LMUX_RECORD=1 uv run pytest --no-cov tests/integration  # record

`--no-cov` must precede the path: the project's `addopts` has a bare `--cov` whose
optional argument would otherwise consume the path as its coverage source.

A test is written once against the `scenario` fixture, which wires the backend for
the active mode; a test declares the modes it supports via markers (`@verified` =
all three).
"""

import json
import math
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lmux.types import ChatChunk, ChatResponse, EmbeddingResponse, ResponseResponse

_LIVE = os.environ.get("LMUX_LIVE") == "1"
_RECORD = os.environ.get("LMUX_RECORD") == "1"

# Exactly one active mode per run: offline replays cassettes, live hits the real
# API, record hits the real API *through the provider* and refreshes the cassette.
_MODE = "record" if _RECORD else ("live" if _LIVE else "offline")
_ALL_MODES = frozenset({"offline", "live", "record"})

_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls"}
_SECRET_KEYS = {"authorization", "api_key", "api-key", "openai-organization", "openai-project", "x-api-key"}


# MARK: Gating


def pytest_configure(config: pytest.Config) -> None:
    for name, desc in (
        ("verified", "sugar for @offline @live @record — runs in all three modes."),
        ("offline", "supports offline mode (cassette replay)."),
        ("live", "supports live mode (real endpoint; LMUX_LIVE=1)."),
        ("record", "supports record mode (refresh cassette from a real call; LMUX_RECORD=1)."),
    ):
        config.addinivalue_line("markers", f"{name}: {desc}")


def _supported_modes(item: pytest.Item) -> frozenset[str]:
    """Which run-modes a test declares support for. ``@verified`` (or no mode
    marker) = all three; explicit ``@offline`` / ``@live`` / ``@record`` = exactly those."""
    if item.get_closest_marker("verified"):
        return _ALL_MODES
    explicit = frozenset(m for m in _ALL_MODES if item.get_closest_marker(m))
    return explicit or _ALL_MODES


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _MODE not in _supported_modes(item):
            item.add_marker(pytest.mark.skip(reason=f"not run in {_MODE!r} mode"))


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


def _scrub(obj: object) -> object:
    """Drop any credential-ish keys before a cassette touches disk."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if str(k).lower() not in _SECRET_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


# MARK: Scenario — one test, three modes (offline / live / record)


class _OfflineAuth:
    """Generic offline auth for key-based providers — the replay transport serves the response, so it's unused."""

    def get_credentials(self) -> str:
        return "offline-not-used"

    async def aget_credentials(self) -> str:
        return "offline-not-used"


_OFFLINE_AUTH = _OfflineAuth()


class _RecordingTransport(httpx.BaseTransport):
    """Forward to the real network and buffer the (request, response) so record mode
    writes the cassette from the provider's *actual* exchange. Buffering the whole
    body works for unary and SSE alike."""

    def __init__(self, sink: list[tuple[httpx.Request, httpx.Response]]) -> None:
        self._inner = httpx.HTTPTransport()
        self._sink = sink

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        inner = self._inner.handle_request(request)
        body = b"".join(inner.stream)
        inner.close()
        self._sink.append((request, httpx.Response(inner.status_code, headers=inner.headers, content=body)))
        return httpx.Response(inner.status_code, headers=inner.headers, content=body, request=request)

    def close(self) -> None:
        self._inner.close()


def _replay_transport(cassette: dict[str, Any], sink: list[httpx.Request]) -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        if "response_sse" in cassette:
            return httpx.Response(
                200, content=cassette["response_sse"].encode(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=cassette["response"])

    return httpx.MockTransport(_handler)


def _write_captured_cassette(cassette_path: Path, request: httpx.Request, response: httpx.Response) -> None:
    endpoint = str(request.url).split("?", 1)[0]
    req_body = json.loads(request.content) if request.content else {}
    if "text/event-stream" in response.headers.get("content-type", ""):
        cassette: dict[str, Any] = {"endpoint": endpoint, "request": _scrub(req_body), "response_sse": response.text}
    else:
        cassette = {"endpoint": endpoint, "request": _scrub(req_body), "response": _scrub(response.json())}
    cassette_path.parent.mkdir(parents=True, exist_ok=True)
    cassette_path.write_text(json.dumps(cassette, indent=2) + "\n")


@pytest.fixture
def scenario() -> Callable[..., Any]:
    """Run a provider call in the active mode and return its response.

    ``scenario(cassette_path, call, requires=...)`` where ``call(auth, transport)``
    builds and invokes the provider with an injected httpx transport: a replay
    transport offline, a recording transport in record mode, and the default (real)
    transport live. offline replays the cassette and asserts the provider called the
    recorded endpoint; live hits the API; record hits the API *through the provider*,
    captures the exchange, and writes the cassette.
    ``requires`` names the env var live/record need (those modes skip if it's absent)."""
    captured: list[tuple[httpx.Request, httpx.Response]] = []
    recorded: set[Path] = set()

    def _run(cassette_path: Path, call: Callable[..., Any], *, requires: str | None = None) -> Any:  # noqa: ANN401
        if _MODE == "offline":
            cassette = json.loads(cassette_path.read_text())
            seen: list[httpx.Request] = []
            resp = call(_OFFLINE_AUTH, _replay_transport(cassette, seen))
            # Assert the outbound endpoint here, not inside the transport: providers wrap
            # request errors in a broad except that would swallow an assertion raised there.
            assert seen, "the provider made no request"
            recorded_path = httpx.URL(cassette["endpoint"]).path
            assert seen[0].url.path == recorded_path, (
                f"replayed request hit {seen[0].url.path}, cassette recorded {recorded_path}"
            )
            return resp
        if requires and not os.environ.get(requires):
            pytest.skip(f"{requires} not set")
        if _MODE == "record":
            start = len(captured)
            resp = call(None, _RecordingTransport(captured))
            if cassette_path not in recorded:
                request, response = captured[start]
                _write_captured_cassette(cassette_path, request, response)
                recorded.add(cassette_path)
            return resp
        return call(None, None)  # live

    return _run


@pytest.fixture
def cosine_similarity() -> Callable[[list[float], list[float]], float]:
    """Cosine similarity of two vectors (1.0 == identical direction)."""

    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    return _cos
