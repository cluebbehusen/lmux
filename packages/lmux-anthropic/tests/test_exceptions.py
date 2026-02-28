"""Tests for Anthropic exception mapping."""

from unittest.mock import MagicMock

import anthropic
import pytest

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_anthropic._exceptions import map_anthropic_error

# MARK: Fixtures


@pytest.fixture
def auth_error() -> anthropic.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return anthropic.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def permission_denied_error() -> anthropic.PermissionDeniedError:
    response = MagicMock()
    response.status_code = 403
    response.headers = {}
    return anthropic.PermissionDeniedError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error() -> anthropic.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return anthropic.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_with_retry() -> anthropic.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "30.5"}
    return anthropic.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_invalid_retry() -> anthropic.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "not-a-number"}
    return anthropic.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def bad_request_error() -> anthropic.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return anthropic.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def not_found_error() -> anthropic.NotFoundError:
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return anthropic.NotFoundError(message="test error", response=response, body=None)


@pytest.fixture
def internal_server_error() -> anthropic.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return anthropic.InternalServerError(message="test error", response=response, body=None)


@pytest.fixture
def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=MagicMock())


# MARK: Tests


class TestMapAnthropicError:
    def test_authentication_error(self, auth_error: anthropic.AuthenticationError) -> None:
        result = map_anthropic_error(auth_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"
        assert result.status_code == 401

    def test_permission_denied_error(self, permission_denied_error: anthropic.PermissionDeniedError) -> None:
        result = map_anthropic_error(permission_denied_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"
        assert result.status_code == 403

    def test_rate_limit_error(self, rate_limit_error: anthropic.RateLimitError) -> None:
        result = map_anthropic_error(rate_limit_error)
        assert isinstance(result, RateLimitError)
        assert result.provider == "anthropic"
        assert result.status_code == 429
        assert result.retry_after is None

    def test_rate_limit_with_retry_after(self, rate_limit_error_with_retry: anthropic.RateLimitError) -> None:
        result = map_anthropic_error(rate_limit_error_with_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after == 30.5

    def test_rate_limit_invalid_retry_after(self, rate_limit_error_invalid_retry: anthropic.RateLimitError) -> None:
        result = map_anthropic_error(rate_limit_error_invalid_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_bad_request_error(self, bad_request_error: anthropic.BadRequestError) -> None:
        result = map_anthropic_error(bad_request_error)
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    def test_not_found_error(self, not_found_error: anthropic.NotFoundError) -> None:
        result = map_anthropic_error(not_found_error)
        assert isinstance(result, NotFoundError)
        assert result.status_code == 404

    def test_internal_server_error(self, internal_server_error: anthropic.InternalServerError) -> None:
        result = map_anthropic_error(internal_server_error)
        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    def test_timeout_error(self, timeout_error: anthropic.APITimeoutError) -> None:
        result = map_anthropic_error(timeout_error)
        assert isinstance(result, TimeoutError)
        assert result.provider == "anthropic"

    def test_generic_api_status_error(self) -> None:
        response = MagicMock()
        response.status_code = 422
        response.headers = {}
        error = anthropic.UnprocessableEntityError(message="test error", response=response, body=None)
        result = map_anthropic_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "anthropic"

    def test_api_connection_error(self) -> None:
        error = anthropic.APIConnectionError(request=MagicMock())
        result = map_anthropic_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "anthropic"
        assert result.status_code is None

    def test_non_anthropic_exception(self) -> None:
        error = RuntimeError("something broke")
        result = map_anthropic_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "anthropic"

    def test_rate_limit_no_response_attribute(self) -> None:
        exc = anthropic.RateLimitError(
            message="rate limited", response=MagicMock(status_code=429, headers={}), body=None
        )
        del exc.response
        result = map_anthropic_error(exc)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                anthropic.AuthenticationError(message="e", response=MagicMock(status_code=401, headers={}), body=None),
                id="auth",
            ),
            pytest.param(
                anthropic.PermissionDeniedError(
                    message="e", response=MagicMock(status_code=403, headers={}), body=None
                ),
                id="permission_denied",
            ),
            pytest.param(
                anthropic.RateLimitError(message="e", response=MagicMock(status_code=429, headers={}), body=None),
                id="rate_limit",
            ),
            pytest.param(
                anthropic.BadRequestError(message="e", response=MagicMock(status_code=400, headers={}), body=None),
                id="bad_request",
            ),
            pytest.param(
                anthropic.NotFoundError(message="e", response=MagicMock(status_code=404, headers={}), body=None),
                id="not_found",
            ),
            pytest.param(
                anthropic.InternalServerError(message="e", response=MagicMock(status_code=500, headers={}), body=None),
                id="internal_server",
            ),
            pytest.param(anthropic.APITimeoutError(request=MagicMock()), id="timeout"),
            pytest.param(RuntimeError("fallback"), id="runtime"),
        ],
    )
    def test_all_mapped_errors_are_lmux_errors(self, error: Exception) -> None:
        result = map_anthropic_error(error)
        assert isinstance(result, LmuxError)
