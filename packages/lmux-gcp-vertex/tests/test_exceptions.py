"""Tests for Google Vertex AI exception mapping."""

import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.genai.errors import APIError, ClientError, ServerError

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_gcp_vertex._exceptions import map_gcp_vertex_error

# MARK: Fixtures


@pytest.fixture
def client_error_400() -> ClientError:
    return ClientError(code=400, response_json={"error": {"message": "bad request"}})


@pytest.fixture
def client_error_401() -> ClientError:
    return ClientError(code=401, response_json={"error": {"message": "unauthenticated"}})


@pytest.fixture
def client_error_403() -> ClientError:
    return ClientError(code=403, response_json={"error": {"message": "permission denied"}})


@pytest.fixture
def client_error_404() -> ClientError:
    return ClientError(code=404, response_json={"error": {"message": "not found"}})


@pytest.fixture
def client_error_408() -> ClientError:
    return ClientError(code=408, response_json={"error": {"message": "timeout"}})


@pytest.fixture
def client_error_429() -> ClientError:
    return ClientError(code=429, response_json={"error": {"message": "rate limited"}})


@pytest.fixture
def client_error_unknown() -> ClientError:
    return ClientError(code=418, response_json={"error": {"message": "teapot"}})


@pytest.fixture
def server_error_500() -> ServerError:
    return ServerError(code=500, response_json={"error": {"message": "internal error"}})


@pytest.fixture
def api_error_502() -> APIError:
    return APIError(code=502, response_json={"error": {"message": "bad gateway"}})


@pytest.fixture
def default_credentials_error() -> DefaultCredentialsError:
    return DefaultCredentialsError("Could not find default credentials")


@pytest.fixture
def refresh_error() -> RefreshError:
    return RefreshError("Token refresh failed")


# MARK: Tests


class TestMapGCPVertexError:
    def test_client_error_400(self, client_error_400: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_400)
        assert isinstance(result, InvalidRequestError)
        assert result.provider == "google"
        assert result.status_code == 400

    def test_client_error_401(self, client_error_401: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_401)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "google"
        assert result.status_code == 401

    def test_client_error_403(self, client_error_403: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_403)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "google"
        assert result.status_code == 403

    def test_client_error_404(self, client_error_404: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_404)
        assert isinstance(result, NotFoundError)
        assert result.provider == "google"
        assert result.status_code == 404

    def test_client_error_408(self, client_error_408: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_408)
        assert isinstance(result, TimeoutError)
        assert result.provider == "google"
        assert result.status_code == 408

    def test_client_error_429(self, client_error_429: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_429)
        assert isinstance(result, RateLimitError)
        assert result.provider == "google"
        assert result.status_code == 429

    def test_client_error_unknown_code(self, client_error_unknown: ClientError) -> None:
        result = map_gcp_vertex_error(client_error_unknown)
        assert isinstance(result, ProviderError)
        assert result.provider == "google"
        assert result.status_code == 418

    def test_server_error(self, server_error_500: ServerError) -> None:
        result = map_gcp_vertex_error(server_error_500)
        assert isinstance(result, ProviderError)
        assert result.provider == "google"
        assert result.status_code == 500

    def test_api_error(self, api_error_502: APIError) -> None:
        result = map_gcp_vertex_error(api_error_502)
        assert isinstance(result, ProviderError)
        assert result.provider == "google"
        assert result.status_code == 502

    def test_default_credentials_error(self, default_credentials_error: DefaultCredentialsError) -> None:
        result = map_gcp_vertex_error(default_credentials_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "google"

    def test_refresh_error(self, refresh_error: RefreshError) -> None:
        result = map_gcp_vertex_error(refresh_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "google"

    def test_generic_exception(self) -> None:
        error = RuntimeError("something broke")
        result = map_gcp_vertex_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "google"

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(ClientError(code=400, response_json={}), id="bad_request"),
            pytest.param(ClientError(code=401, response_json={}), id="unauthenticated"),
            pytest.param(ClientError(code=403, response_json={}), id="permission_denied"),
            pytest.param(ClientError(code=404, response_json={}), id="not_found"),
            pytest.param(ClientError(code=408, response_json={}), id="timeout"),
            pytest.param(ClientError(code=429, response_json={}), id="rate_limited"),
            pytest.param(ClientError(code=418, response_json={}), id="unknown_client"),
            pytest.param(ServerError(code=500, response_json={}), id="server"),
            pytest.param(DefaultCredentialsError(""), id="no_credentials"),
            pytest.param(RefreshError(""), id="refresh"),
            pytest.param(RuntimeError("fallback"), id="runtime"),
        ],
    )
    def test_all_errors_are_lmux_errors(self, error: Exception) -> None:
        result = map_gcp_vertex_error(error)
        assert isinstance(result, LmuxError)
        assert result.provider == "google"
