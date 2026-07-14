"""Google implicit context caching on Vertex AI: a large fixed prompt sent twice back-to-back
produces a cache read on the second call, billed at the published cache-read rate.

Implicit caching is best-effort (no explicit cache create), so this runs offline and record
only — never live. Record captures a real cold-then-warm pair; offline replays it.
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, UserMessage

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 256
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "google"
_WRITE_CASSETTE = _CASSETTES / "vertex_cache_cold.json"
_READ_CASSETTE = _CASSETTES / "vertex_cache_warm.json"

# A ~26k-token prompt so the identical second call hits the implicit cache — gemini-2.5-flash only
# caches reliably well above its nominal floor (empirically ~24k tokens). The unique prefix makes the
# first call genuinely cold at record time (the server-side cache persists across runs), while both
# calls in a run share it so the second is a warm read. Offline ignores the body, so the nonce is inert.
_NONCE = uuid.uuid4().hex
_FILLER = " ".join(
    f"Sentence {i}: the quick brown fox jumps over the lazy dog while the sun sets." for i in range(1300)
)
_PROMPT = f"{_NONCE}\n\n{_FILLER}\n\nBased on the text above, reply with exactly the word: pong"

_RATES = {"input_rate": 0.30 / 1_000_000, "output_rate": 2.50 / 1_000_000, "cache_read_rate": 0.03 / 1_000_000}


@pytest.mark.offline
@pytest.mark.record
def test_cold_write_then_warm_read(
    scenario: Callable[..., Any],
    vertex_provider: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
) -> None:
    def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
        return vertex_provider(auth, transport).chat(_MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS)

    cold = scenario(_WRITE_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert_chat(cold, provider="google")
    assert cold.usage is not None
    assert cold.usage.cache_read_tokens is None  # a cold call reads nothing from cache
    assert_cost(cold, **_RATES)

    warm = scenario(_READ_CASSETTE, _chat, requires="VERTEXAI_API_KEY")
    assert_chat(warm, provider="google")
    assert warm.usage is not None
    read = warm.usage.cache_read_tokens
    assert read is not None
    assert read > 2048  # the fixed prompt is well over the implicit-cache floor
    assert_cost(warm, **_RATES)
