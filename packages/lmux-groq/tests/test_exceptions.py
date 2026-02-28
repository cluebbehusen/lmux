"""Tests for Groq exception mapping."""

from unittest.mock import MagicMock

import groq
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
from lmux_groq._exceptions import map_groq_error

# MARK: Fixtures


@pytest.fixture
def auth_error() -> groq.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return groq.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def permission_denied_error() -> groq.PermissionDeniedError:
    response = MagicMock()
    response.status_code = 403
    response.headers = {}
    return groq.PermissionDeniedError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error() -> groq.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return groq.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_with_retry() -> groq.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "30.5"}
    return groq.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_invalid_retry() -> groq.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "not-a-number"}
    return groq.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def bad_request_error() -> groq.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return groq.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def not_found_error() -> groq.NotFoundError:
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return groq.NotFoundError(message="test error", response=response, body=None)


@pytest.fixture
def internal_server_error() -> groq.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return groq.InternalServerError(message="test error", response=response, body=None)


@pytest.fixture
def timeout_error() -> groq.APITimeoutError:
    return groq.APITimeoutError(request=MagicMock())


@pytest.fixture
def unprocessable_error() -> groq.UnprocessableEntityError:
    response = MagicMock()
    response.status_code = 422
    response.headers = {}
    return groq.UnprocessableEntityError(message="test error", response=response, body=None)


# MARK: Tests


class TestMapGroqError:
    def test_authentication_error(self, auth_error: groq.AuthenticationError) -> None:
        result = map_groq_error(auth_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "groq"
        assert result.status_code == 401

    def test_permission_denied_error(self, permission_denied_error: groq.PermissionDeniedError) -> None:
        result = map_groq_error(permission_denied_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "groq"
        assert result.status_code == 403

    def test_rate_limit_error(self, rate_limit_error: groq.RateLimitError) -> None:
        result = map_groq_error(rate_limit_error)
        assert isinstance(result, RateLimitError)
        assert result.provider == "groq"
        assert result.status_code == 429
        assert result.retry_after is None

    def test_rate_limit_with_retry_after(self, rate_limit_error_with_retry: groq.RateLimitError) -> None:
        result = map_groq_error(rate_limit_error_with_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after == 30.5

    def test_rate_limit_invalid_retry_after(self, rate_limit_error_invalid_retry: groq.RateLimitError) -> None:
        result = map_groq_error(rate_limit_error_invalid_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_bad_request_error(self, bad_request_error: groq.BadRequestError) -> None:
        result = map_groq_error(bad_request_error)
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    def test_not_found_error(self, not_found_error: groq.NotFoundError) -> None:
        result = map_groq_error(not_found_error)
        assert isinstance(result, NotFoundError)
        assert result.status_code == 404

    def test_internal_server_error(self, internal_server_error: groq.InternalServerError) -> None:
        result = map_groq_error(internal_server_error)
        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    def test_timeout_error(self, timeout_error: groq.APITimeoutError) -> None:
        result = map_groq_error(timeout_error)
        assert isinstance(result, TimeoutError)
        assert result.provider == "groq"

    def test_generic_api_status_error(self, unprocessable_error: groq.UnprocessableEntityError) -> None:
        result = map_groq_error(unprocessable_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "groq"

    def test_api_connection_error(self) -> None:
        error = groq.APIConnectionError(request=MagicMock())
        result = map_groq_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "groq"
        assert result.status_code is None

    def test_non_groq_exception(self) -> None:
        error = RuntimeError("something broke")
        result = map_groq_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "groq"

    def test_rate_limit_no_response_attribute(self) -> None:
        exc = groq.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None)
        del exc.response
        result = map_groq_error(exc)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                groq.AuthenticationError(message="e", response=MagicMock(status_code=401, headers={}), body=None),
                id="auth",
            ),
            pytest.param(
                groq.PermissionDeniedError(message="e", response=MagicMock(status_code=403, headers={}), body=None),
                id="permission_denied",
            ),
            pytest.param(
                groq.RateLimitError(message="e", response=MagicMock(status_code=429, headers={}), body=None),
                id="rate_limit",
            ),
            pytest.param(
                groq.BadRequestError(message="e", response=MagicMock(status_code=400, headers={}), body=None),
                id="bad_request",
            ),
            pytest.param(
                groq.NotFoundError(message="e", response=MagicMock(status_code=404, headers={}), body=None),
                id="not_found",
            ),
            pytest.param(
                groq.InternalServerError(message="e", response=MagicMock(status_code=500, headers={}), body=None),
                id="internal_server",
            ),
            pytest.param(groq.APITimeoutError(request=MagicMock()), id="timeout"),
            pytest.param(RuntimeError("fallback"), id="runtime"),
        ],
    )
    def test_all_mapped_errors_are_lmux_errors(self, error: Exception) -> None:
        result = map_groq_error(error)
        assert isinstance(result, LmuxError)
