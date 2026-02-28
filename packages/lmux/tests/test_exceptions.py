"""Tests for lmux exception hierarchy."""

import pytest

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
    UnsupportedFeatureError,
)


class TestLmuxError:
    def test_message(self) -> None:
        error = LmuxError("something went wrong")
        assert str(error) == "something went wrong"

    def test_default_attrs(self) -> None:
        error = LmuxError("fail")
        assert error.provider is None
        assert error.status_code is None

    def test_custom_attrs(self) -> None:
        error = LmuxError("fail", provider="openai", status_code=500)
        assert error.provider == "openai"
        assert error.status_code == 500


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_class",
        [
            AuthenticationError,
            RateLimitError,
            InvalidRequestError,
            NotFoundError,
            ProviderError,
            TimeoutError,
            UnsupportedFeatureError,
        ],
    )
    def test_subclass_of_lmux_error(self, exc_class: type[LmuxError]) -> None:
        assert issubclass(exc_class, LmuxError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            AuthenticationError,
            RateLimitError,
            InvalidRequestError,
            NotFoundError,
            ProviderError,
            TimeoutError,
            UnsupportedFeatureError,
        ],
    )
    def test_catchable_as_lmux_error(self, exc_class: type[LmuxError]) -> None:
        with pytest.raises(LmuxError):
            raise exc_class("test", provider="test_provider", status_code=400)

    @pytest.mark.parametrize(
        "exc_class",
        [
            AuthenticationError,
            InvalidRequestError,
            NotFoundError,
            ProviderError,
            TimeoutError,
            UnsupportedFeatureError,
        ],
    )
    def test_inherits_attrs(self, exc_class: type[LmuxError]) -> None:
        error = exc_class("msg", provider="p", status_code=123)
        assert error.provider == "p"
        assert error.status_code == 123


class TestRateLimitError:
    def test_retry_after(self) -> None:
        error = RateLimitError("rate limited", retry_after=30.0)
        assert error.retry_after == 30.0

    def test_retry_after_default(self) -> None:
        error = RateLimitError("rate limited")
        assert error.retry_after is None

    def test_all_attrs(self) -> None:
        error = RateLimitError("rate limited", provider="openai", status_code=429, retry_after=5.0)
        assert error.provider == "openai"
        assert error.status_code == 429
        assert error.retry_after == 5.0
