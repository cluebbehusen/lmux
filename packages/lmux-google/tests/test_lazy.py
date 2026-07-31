"""Tests for the Google HTTP client factories, endpoint resolution, and Vertex auth glue."""

from typing import TYPE_CHECKING, cast

import httpx
import respx

from lmux_google._lazy import (
    HttpxAuthRequest,
    api_key_headers,
    bearer_headers,
    bearer_token,
    vertex_base_url,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


class _Credentials:
    def __init__(self, *, valid: bool, token: str | None = "live-token") -> None:  # noqa: S107
        self.valid = valid
        self.token = token
        self.refreshed = False

    def refresh(self, request: object) -> None:  # noqa: ARG002
        self.refreshed = True
        self.valid = True
        self.token = "refreshed"  # noqa: S105


class TestVertexBaseUrl:
    def test_regional(self) -> None:
        assert vertex_base_url("us-central1") == "https://us-central1-aiplatform.googleapis.com"

    def test_global(self) -> None:
        assert vertex_base_url("global") == "https://aiplatform.googleapis.com"

    def test_none(self) -> None:
        assert vertex_base_url(None) == "https://aiplatform.googleapis.com"

    def test_us_multi_region(self) -> None:
        assert vertex_base_url("us") == "https://aiplatform.us.rep.googleapis.com"

    def test_eu_multi_region(self) -> None:
        assert vertex_base_url("eu") == "https://aiplatform.eu.rep.googleapis.com"


class TestApiKeyHeaders:
    def test_builds_headers(self) -> None:
        assert api_key_headers("secret") == {"x-goog-api-key": "secret", "Content-Type": "application/json"}

    def test_merges_defaults_case_insensitively(self) -> None:
        assert api_key_headers(
            "secret", {"X-Trace-ID": "trace-123", "X-GOOG-API-KEY": "wrong", "content-type": "text/plain"}
        ) == {"X-Trace-ID": "trace-123", "x-goog-api-key": "secret", "Content-Type": "application/json"}


class TestBearerHeaders:
    def test_builds_headers(self) -> None:
        assert bearer_headers("tok") == {"Authorization": "Bearer tok", "Content-Type": "application/json"}

    def test_includes_quota_project(self) -> None:
        assert bearer_headers("tok", "quota-proj") == {
            "Authorization": "Bearer tok",
            "Content-Type": "application/json",
            "x-goog-user-project": "quota-proj",
        }


class TestBearerToken:
    def test_returns_valid_token_without_refresh(self) -> None:
        creds = _Credentials(valid=True)
        assert bearer_token(cast("Credentials", creds)) == "live-token"
        assert creds.refreshed is False

    def test_refreshes_when_invalid(self) -> None:
        creds = _Credentials(valid=False)
        assert bearer_token(cast("Credentials", creds)) == "refreshed"
        assert creds.refreshed is True


class TestHttpxAuthRequest:
    def test_performs_http_request(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://oauth2.example/token").mock(
            return_value=httpx.Response(200, json={"access_token": "t"}, headers={"x-test": "1"})
        )
        response = HttpxAuthRequest()(
            "https://oauth2.example/token",
            method="POST",
            body=b"grant=x",
            headers={"content-type": "text/plain"},
            timeout=5.0,
        )
        assert response.status == 200
        assert response.headers["x-test"] == "1"
        assert b"access_token" in response.data
        assert route.called

    def test_default_get(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://oauth2.example/ping").mock(return_value=httpx.Response(204))
        response = HttpxAuthRequest()("https://oauth2.example/ping")
        assert response.status == 204
