"""Which failures the load balancer treats as failover-worthy."""

from lmux import ProviderError, RateLimitError, TimeoutError  # noqa: A004


def is_retryable(exc: Exception) -> bool:
    """Whether a failed endpoint should fall through to the next candidate.

    Retryable: rate limits, timeouts, and provider/server errors (5xx, 529, and
    connection failures, which lmux surfaces as ``ProviderError`` with no status code).
    Everything else (auth, invalid request, not found, unsupported feature) would fail
    identically on every endpoint, so it propagates instead of triggering failover.
    """
    return isinstance(exc, (RateLimitError, TimeoutError, ProviderError))
