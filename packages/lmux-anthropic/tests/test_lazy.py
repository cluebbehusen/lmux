"""Tests for the Anthropic HTTP client factories and Vertex/Foundry/WIF auth glue."""

import json
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import respx

from lmux.exceptions import AuthenticationError, ProviderError, RateLimitError
from lmux_anthropic._lazy import (
    HttpxTransportRequest,
    _resolve_foundry_base_url,
    bearer_auth_headers,
    create_sync_foundry_client,
    create_sync_vertex_client,
    exchange_workload_identity_token,
    foundry_auth_headers,
    foundry_base_url,
    resolve_vertex_token,
    vertex_auth_headers,
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


class TestVertexAuthHeaders:
    def test_builds_bearer(self) -> None:
        creds = _Credentials(access="live-token")
        assert vertex_auth_headers(cast("Credentials", creds)) == {"Authorization": "Bearer live-token"}

    def test_refreshes_expired_token(self) -> None:
        creds = _Credentials(access="stale", expired=True)
        assert vertex_auth_headers(cast("Credentials", creds)) == {"Authorization": "Bearer refreshed"}
        assert creds.refreshed is True


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

    def test_multi_region_us(self) -> None:
        assert vertex_base_url("us") == "https://aiplatform.us.rep.googleapis.com/v1"

    def test_multi_region_eu(self) -> None:
        assert vertex_base_url("eu") == "https://aiplatform.eu.rep.googleapis.com/v1"

    def test_factory_uses_base_url_override(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post("https://vx.test/v1/messages").mock(return_value=httpx.Response(200, json={}))
        client = create_sync_vertex_client(region="us-east5", base_url="https://vx.test")
        response = client.post("v1/messages", json={})
        assert response.status_code == 200


class TestFoundryAuthHeaders:
    def test_api_key_sends_both(self) -> None:
        assert foundry_auth_headers("k", None) == {"x-api-key": "k", "api-key": "k"}

    def test_token_provider(self) -> None:
        assert foundry_auth_headers(None, lambda: "tok") == {"Authorization": "Bearer tok"}

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            foundry_auth_headers(None, None)


class TestFoundryBaseUrl:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)

    def test_from_resource(self) -> None:
        assert foundry_base_url("res") == "https://res.services.ai.azure.com/anthropic"

    def test_base_url_and_resource_conflict(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_foundry_base_url("https://custom", "res")

    def test_explicit_resource_conflicts_with_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A stale env base URL must not silently override an explicitly selected resource.
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://env.foundry")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_foundry_base_url(None, "res")

    def test_from_resource_when_no_base_url(self) -> None:
        assert _resolve_foundry_base_url(None, "res") == "https://res.services.ai.azure.com/anthropic"

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://env.foundry")
        assert _resolve_foundry_base_url(None, None) == "https://env.foundry"

    def test_resource_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "env-res")
        assert _resolve_foundry_base_url(None, None) == "https://env-res.services.ai.azure.com/anthropic"

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url or resource"):
            _resolve_foundry_base_url(None, None)

    def test_factory_with_base_url(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://custom.foundry/v1/messages").mock(return_value=httpx.Response(200, json={}))
        client = create_sync_foundry_client(base_url="https://custom.foundry")
        client.post("v1/messages", json={})
        assert route.called


class TestBearerAuthHeaders:
    def test_invokes_provider_and_adds_oauth_beta(self) -> None:
        assert bearer_auth_headers(lambda: "tok-1") == {
            "Authorization": "Bearer tok-1",
            "anthropic-beta": "oauth-2025-04-20",
        }

    def test_merges_caller_beta_flags(self) -> None:
        headers = bearer_auth_headers(lambda: "tok-1", {"ANTHROPIC-BETA": "context-1m-2025-08-07"})
        assert headers["anthropic-beta"] == "context-1m-2025-08-07,oauth-2025-04-20"

    def test_oauth_beta_not_duplicated(self) -> None:
        headers = bearer_auth_headers(lambda: "tok-1", {"anthropic-beta": "oauth-2025-04-20"})
        assert headers["anthropic-beta"] == "oauth-2025-04-20"


_EXCHANGE_URL = "https://api.anthropic.com/v1/oauth/token"


def _exchange(
    *,
    service_account_id: str | None = "svac_1",
    workspace_id: str | None = None,
    base_url: str | None = None,
) -> tuple[str, float]:
    return exchange_workload_identity_token(
        assertion="id-jwt",
        federation_rule_id="fdrl_1",
        organization_id="org-1",
        service_account_id=service_account_id,
        workspace_id=workspace_id,
        base_url=base_url,
    )


class TestExchangeWorkloadIdentityToken:
    def test_exchanges_and_returns_token(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc", "expires_in": 600})
        )
        assert _exchange() == ("sk-ant-oat01-abc", 600.0)
        request = route.calls.last.request
        assert request.headers["anthropic-beta"] == "oauth-2025-04-20,oidc-federation-2026-04-01"
        assert json.loads(request.content) == {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": "id-jwt",
            "federation_rule_id": "fdrl_1",
            "organization_id": "org-1",
            "service_account_id": "svac_1",
        }

    def test_service_account_id_omitted_when_none(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc", "expires_in": 600})
        )
        _exchange(service_account_id=None)
        assert "service_account_id" not in json.loads(route.calls.last.request.content)

    def test_workspace_id_included_when_set(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc", "expires_in": 600})
        )
        _exchange(workspace_id="wrkspc_1")
        assert json.loads(route.calls.last.request.content)["workspace_id"] == "wrkspc_1"

    def test_base_url_override(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://gateway.example/v1/oauth/token").mock(
            return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc", "expires_in": 600})
        )
        _exchange(base_url="https://gateway.example")
        assert route.called

    def test_base_url_trailing_slash_normalized(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://gateway.example/v1/oauth/token").mock(
            return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc", "expires_in": 600})
        )
        _exchange(base_url="https://gateway.example/")
        assert route.calls.last.request.url.path == "/v1/oauth/token"

    def test_error_response_raises_with_api_message(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(401, json={"error": {"message": "Authentication failed"}})
        )
        with pytest.raises(AuthenticationError, match="token exchange failed: Authentication failed") as exc_info:
            _exchange()
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 401

    def test_rate_limit_status_raises_retryable_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}})
        )
        with pytest.raises(RateLimitError, match="token exchange failed: Rate limited") as exc_info:
            _exchange()
        assert exc_info.value.status_code == 429

    def test_server_error_status_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(return_value=httpx.Response(503, json={"error": {"message": "Overloaded"}}))
        with pytest.raises(ProviderError, match="token exchange failed: Overloaded") as exc_info:
            _exchange()
        assert exc_info.value.status_code == 503

    def test_error_body_not_propagated_verbatim(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(return_value=httpx.Response(502, text="<html>gateway-echo</html>"))
        with pytest.raises(ProviderError, match="token exchange failed") as exc_info:
            _exchange()
        assert "gateway-echo" not in str(exc_info.value)
        assert exc_info.value.status_code == 502

    def test_error_without_message_uses_generic(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(
            return_value=httpx.Response(400, json={"error": {"type": "invalid_request_error"}})
        )
        with pytest.raises(AuthenticationError, match=r"token exchange failed$"):
            _exchange()

    def test_error_message_is_bounded(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "x" * 500}}))
        with pytest.raises(AuthenticationError) as exc_info:
            _exchange()
        assert len(str(exc_info.value)) < 300

    def test_missing_access_token_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(return_value=httpx.Response(200, json={"expires_in": 600}))
        with pytest.raises(AuthenticationError, match="missing access_token or expires_in"):
            _exchange()

    def test_missing_expires_in_raises(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EXCHANGE_URL).mock(return_value=httpx.Response(200, json={"access_token": "sk-ant-oat01-abc"}))
        with pytest.raises(AuthenticationError, match="missing access_token or expires_in"):
            _exchange()
