"""Tests for the Anthropic HTTP client factories and Vertex refresh glue."""

from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lmux_anthropic._lazy import (
    HttpxTransportRequest,
    _foundry_headers,
    _resolve_foundry_base_url,
    create_sync_foundry_client,
    create_sync_vertex_client,
    foundry_base_url,
    resolve_vertex_token,
    vertex_base_url,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials


class _Credentials:
    def __init__(
        self, *, access: str | None, expired: bool = False, refreshed_access: str | None = "refreshed"
    ) -> None:
        self.token = access
        self.expired = expired
        self._refreshed_access = refreshed_access
        self.refreshed = False

    def refresh(self, request: object) -> None:  # noqa: ARG002
        self.refreshed = True
        self.token = self._refreshed_access


class TestResolveVertexToken:
    def test_returns_existing_token_without_refresh(self) -> None:
        creds = _Credentials(access="live-token")
        assert resolve_vertex_token(cast("Credentials", creds)) == "live-token"
        assert creds.refreshed is False

    def test_refreshes_when_no_token(self) -> None:
        creds = _Credentials(access=None)
        assert resolve_vertex_token(cast("Credentials", creds)) == "refreshed"
        assert creds.refreshed is True

    def test_refreshes_when_expired(self) -> None:
        creds = _Credentials(access="stale", expired=True)
        assert resolve_vertex_token(cast("Credentials", creds)) == "refreshed"
        assert creds.refreshed is True

    def test_raises_when_still_no_token(self) -> None:
        creds = _Credentials(access=None, refreshed_access=None)
        with pytest.raises(RuntimeError, match="access token"):
            resolve_vertex_token(cast("Credentials", creds))


class TestHttpxTransportRequest:
    def test_performs_http_request(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://oauth2.example/token").mock(
            return_value=httpx.Response(200, json={"access_token": "t"}, headers={"x-test": "1"})
        )
        response = HttpxTransportRequest()(
            "https://oauth2.example/token", method="POST", body=b"grant=x", headers={"content-type": "text/plain"}
        )
        assert response.status == 200
        assert response.headers["x-test"] == "1"
        assert b"access_token" in response.data
        assert route.called

    def test_default_get(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://oauth2.example/ping").mock(return_value=httpx.Response(204))
        response = HttpxTransportRequest()("https://oauth2.example/ping")
        assert response.status == 204


class TestVertexBaseUrl:
    def test_regional(self) -> None:
        assert vertex_base_url("us-east5") == "https://us-east5-aiplatform.googleapis.com/v1"

    def test_global(self) -> None:
        assert vertex_base_url("global") == "https://aiplatform.googleapis.com/v1"

    def test_factory_uses_base_url_override(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post("https://vx.test/v1/messages").mock(return_value=httpx.Response(200, json={}))
        client = create_sync_vertex_client(
            credentials=cast("Credentials", _Credentials(access="t")), region="us-east5", base_url="https://vx.test"
        )
        response = client.post("v1/messages", json={})
        assert response.status_code == 200


class TestFoundryHeaders:
    def test_api_key(self) -> None:
        assert _foundry_headers("k", None)["api-key"] == "k"

    def test_token_provider(self) -> None:
        assert _foundry_headers(None, lambda: "tok")["Authorization"] == "Bearer tok"

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            _foundry_headers(None, None)


class TestFoundryBaseUrl:
    def test_from_resource(self) -> None:
        assert foundry_base_url("res") == "https://res.services.ai.azure.com/anthropic"

    def test_base_url_wins(self) -> None:
        assert _resolve_foundry_base_url("https://custom", "res") == "https://custom"

    def test_from_resource_when_no_base_url(self) -> None:
        assert _resolve_foundry_base_url(None, "res") == "https://res.services.ai.azure.com/anthropic"

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            _resolve_foundry_base_url(None, None)

    def test_factory_with_base_url(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://custom.foundry/v1/messages").mock(return_value=httpx.Response(200, json={}))
        client = create_sync_foundry_client(api_key="k", base_url="https://custom.foundry")
        client.post("v1/messages", json={})
        assert route.called
