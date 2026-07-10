"""Tests for the retryable-error predicate."""

from lmux import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
    UnsupportedFeatureError,
)
from lmux_load_balancer._retryable import is_retryable


class TestIsRetryable:
    def test_retryable(self) -> None:
        assert is_retryable(RateLimitError("rate"))
        assert is_retryable(TimeoutError("timeout"))
        assert is_retryable(ProviderError("server"))

    def test_not_retryable(self) -> None:
        assert not is_retryable(AuthenticationError("auth"))
        assert not is_retryable(InvalidRequestError("bad"))
        assert not is_retryable(NotFoundError("missing"))
        assert not is_retryable(UnsupportedFeatureError("nope"))
