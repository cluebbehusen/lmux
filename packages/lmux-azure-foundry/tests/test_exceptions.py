"""Tests for Azure AI Foundry exception mapping."""

from unittest.mock import MagicMock

import openai
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
from lmux_azure_foundry._exceptions import map_azure_foundry_error

# MARK: Fixtures


@pytest.fixture
def auth_error() -> openai.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return openai.AuthenticationError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return openai.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_with_retry() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "30.5"}
    return openai.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def rate_limit_error_invalid_retry() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "not-a-number"}
    return openai.RateLimitError(message="test error", response=response, body=None)


@pytest.fixture
def bad_request_error() -> openai.BadRequestError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return openai.BadRequestError(message="test error", response=response, body=None)


@pytest.fixture
def not_found_error() -> openai.NotFoundError:
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return openai.NotFoundError(message="test error", response=response, body=None)


@pytest.fixture
def internal_server_error() -> openai.InternalServerError:
    response = MagicMock()
    response.status_code = 500
    response.headers = {}
    return openai.InternalServerError(message="test error", response=response, body=None)


@pytest.fixture
def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=MagicMock())


@pytest.fixture
def unprocessable_error() -> openai.UnprocessableEntityError:
    response = MagicMock()
    response.status_code = 422
    response.headers = {}
    return openai.UnprocessableEntityError(message="test error", response=response, body=None)


# MARK: Tests


class TestMapAzureFoundryError:
    def test_authentication_error(self, auth_error: openai.AuthenticationError) -> None:
        result = map_azure_foundry_error(auth_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "azure-foundry"
        assert result.status_code == 401

    def test_rate_limit_error(self, rate_limit_error: openai.RateLimitError) -> None:
        result = map_azure_foundry_error(rate_limit_error)
        assert isinstance(result, RateLimitError)
        assert result.provider == "azure-foundry"
        assert result.status_code == 429
        assert result.retry_after is None

    def test_rate_limit_with_retry_after(self, rate_limit_error_with_retry: openai.RateLimitError) -> None:
        result = map_azure_foundry_error(rate_limit_error_with_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after == 30.5

    def test_rate_limit_invalid_retry_after(self, rate_limit_error_invalid_retry: openai.RateLimitError) -> None:
        result = map_azure_foundry_error(rate_limit_error_invalid_retry)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_bad_request_error(self, bad_request_error: openai.BadRequestError) -> None:
        result = map_azure_foundry_error(bad_request_error)
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    def test_not_found_error(self, not_found_error: openai.NotFoundError) -> None:
        result = map_azure_foundry_error(not_found_error)
        assert isinstance(result, NotFoundError)
        assert result.status_code == 404

    def test_internal_server_error(self, internal_server_error: openai.InternalServerError) -> None:
        result = map_azure_foundry_error(internal_server_error)
        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    def test_timeout_error(self, timeout_error: openai.APITimeoutError) -> None:
        result = map_azure_foundry_error(timeout_error)
        assert isinstance(result, TimeoutError)
        assert result.provider == "azure-foundry"

    def test_generic_api_error(self, unprocessable_error: openai.UnprocessableEntityError) -> None:
        result = map_azure_foundry_error(unprocessable_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "azure-foundry"

    def test_non_openai_exception(self) -> None:
        error = RuntimeError("something broke")
        result = map_azure_foundry_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "azure-foundry"

    def test_rate_limit_no_response_attribute(self) -> None:
        """Exercise _extract_retry_after when error has no response."""
        exc = openai.RateLimitError(message="rate limited", response=MagicMock(status_code=429, headers={}), body=None)
        del exc.response
        result = map_azure_foundry_error(exc)
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                openai.AuthenticationError(message="e", response=MagicMock(status_code=401, headers={}), body=None),
                id="auth",
            ),
            pytest.param(
                openai.RateLimitError(message="e", response=MagicMock(status_code=429, headers={}), body=None),
                id="rate_limit",
            ),
            pytest.param(
                openai.BadRequestError(message="e", response=MagicMock(status_code=400, headers={}), body=None),
                id="bad_request",
            ),
            pytest.param(
                openai.NotFoundError(message="e", response=MagicMock(status_code=404, headers={}), body=None),
                id="not_found",
            ),
            pytest.param(
                openai.InternalServerError(message="e", response=MagicMock(status_code=500, headers={}), body=None),
                id="internal_server",
            ),
            pytest.param(openai.APITimeoutError(request=MagicMock()), id="timeout"),
            pytest.param(RuntimeError("fallback"), id="runtime"),
        ],
    )
    def test_all_mapped_errors_are_lmux_errors(self, error: Exception) -> None:
        result = map_azure_foundry_error(error)
        assert isinstance(result, LmuxError)
