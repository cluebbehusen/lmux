"""Tests for Anthropic auth providers."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, sentinel

import boto3
import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import AuthenticationError
from lmux_anthropic.auth import (
    AnthropicBedrockEnvAuthProvider,
    AnthropicBedrockSessionAuthProvider,
    AnthropicEnvAuthProvider,
    AnthropicFoundryEnvAuthProvider,
    AnthropicFoundryTokenAuthProvider,
    AnthropicVertexADCAuthProvider,
    AnthropicVertexServiceAccountAuthProvider,
    AnthropicWorkloadIdentityAuthProvider,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@pytest.fixture
def mock_credentials() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_google_auth_default(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("google.auth.default", return_value=(mock_credentials, "test-project"))


@pytest.fixture
def mock_load_credentials(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "google.auth.load_credentials_from_file",
        return_value=(mock_credentials, "file-project"),
    )


@pytest.fixture
def mock_with_scopes(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("google.auth.credentials.with_scopes_if_required", return_value=mock_credentials)


@pytest.fixture
def mock_cloud_sdk_path(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "google.auth._cloud_sdk.get_application_default_credentials_path",
        return_value="/missing/gcloud/application_default_credentials.json",
    )


@pytest.fixture
def mock_cloud_sdk_project(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("google.auth._cloud_sdk.get_project_id", return_value="gcloud-project")


@pytest.fixture
def mock_transport_request(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic._lazy.HttpxTransportRequest", return_value=sentinel.request)


@pytest.fixture
def mock_from_service_account_file(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    mock_credentials.project_id = "sa-project"
    return mocker.patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=mock_credentials,
    )


@pytest.fixture
def mock_missing_google_auth(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"google.auth": None})


@pytest.fixture
def mock_missing_google_oauth2(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"google.oauth2": None})


@pytest.fixture(autouse=True)
def clear_adc_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)


class TestAnthropicEnvAuthProvider:
    def test_get_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        provider = AnthropicEnvAuthProvider()
        assert provider.get_credentials() == "sk-ant-test-key"

    def test_get_credentials_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="ANTHROPIC_API_KEY") as exc_info:
            _ = provider.get_credentials()
        assert exc_info.value.provider == "anthropic"

    async def test_aget_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        provider = AnthropicEnvAuthProvider()
        assert await provider.aget_credentials() == "sk-ant-test-key"


@pytest.fixture
def mock_exchange(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "lmux_anthropic.auth.exchange_workload_identity_token",
        return_value=("sk-ant-oat01-1", 600.0),
    )


def _wif_provider(
    *,
    identity_token_provider: Callable[[], str] | None = None,
    identity_token_file: str | Path | None = None,
    workspace_id: str | None = None,
    token_base_url: str | None = None,
) -> AnthropicWorkloadIdentityAuthProvider:
    return AnthropicWorkloadIdentityAuthProvider(
        federation_rule_id="fdrl_1",
        organization_id="org-1",
        service_account_id="svac_1",
        workspace_id=workspace_id,
        identity_token_provider=identity_token_provider,
        identity_token_file=identity_token_file,
        token_base_url=token_base_url,
    )


class TestAnthropicWorkloadIdentityAuthProvider:
    def test_get_credentials_exchanges_identity_token(self, mock_exchange: MagicMock) -> None:
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        mock_exchange.assert_called_once_with(
            assertion="id-jwt",
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            service_account_id="svac_1",
            workspace_id=None,
            base_url=None,
        )

    def test_workspace_id_and_token_base_url_forwarded(self, mock_exchange: MagicMock) -> None:
        provider = _wif_provider(
            identity_token_provider=lambda: "id-jwt",
            workspace_id="wrkspc_1",
            token_base_url="https://gateway.example",  # noqa: S106
        )
        provider.get_credentials()()
        mock_exchange.assert_called_once_with(
            assertion="id-jwt",
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            service_account_id="svac_1",
            workspace_id="wrkspc_1",
            base_url="https://gateway.example",
        )

    def test_access_token_cached_until_refresh(self, mock_exchange: MagicMock) -> None:
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        assert token_provider() == "sk-ant-oat01-1"
        mock_exchange.assert_called_once()

    def test_reexchanges_fresh_identity_token_near_expiry(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = [0.0]
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: clock[0])
        identity_tokens = iter(["id-jwt-1", "id-jwt-2"])
        mock_exchange.side_effect = [("sk-ant-oat01-1", 600.0), ("sk-ant-oat01-2", 600.0)]
        provider = _wif_provider(identity_token_provider=lambda: next(identity_tokens))
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        clock[0] = 540.0  # 600s lifetime minus the 60s refresh leeway
        assert token_provider() == "sk-ant-oat01-2"
        assert mock_exchange.call_count == 2
        mock_exchange.assert_called_with(
            assertion="id-jwt-2",
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            service_account_id="svac_1",
            workspace_id=None,
            base_url=None,
        )

    def test_expired_waiter_reuses_token_refreshed_while_locked(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clock sequence: the pre-lock check sees the token as expired, but the re-check under
        # the lock sees it fresh — the state a concurrent winner publishes while this caller
        # waits on the lock — so no second exchange happens.
        clocks = iter([0.0, 600.0, 0.0])
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: next(clocks))
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        assert token_provider() == "sk-ant-oat01-1"
        mock_exchange.assert_called_once()

    def test_advisory_recheck_skips_when_already_refreshed(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clock sequence: the advisory check sees the refresh point passed, but the re-check
        # under the try-lock sees it fresh — the state a concurrent winner published — so the
        # cached token is served without another exchange.
        clocks = iter([0.0, 550.0, 550.0, 0.0])
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: next(clocks))
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        assert token_provider() == "sk-ant-oat01-1"
        mock_exchange.assert_called_once()

    def test_advisory_refresh_skipped_when_lock_held(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = [0.0]
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: clock[0])
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        clock[0] = 550.0  # in the advisory window while another caller's refresh is in flight
        provider._refresh_lock.acquire()
        try:
            assert token_provider() == "sk-ant-oat01-1"
        finally:
            provider._refresh_lock.release()
        mock_exchange.assert_called_once()

    def test_refresh_failure_in_advisory_window_serves_cached_token(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = [0.0]
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: clock[0])
        mock_exchange.side_effect = [
            ("sk-ant-oat01-1", 600.0),
            AuthenticationError("boom", provider="anthropic"),
        ]
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        clock[0] = 550.0  # past the 540s advisory refresh, before the 600s expiry
        assert token_provider() == "sk-ant-oat01-1"
        assert mock_exchange.call_count == 2

    def test_refresh_failure_after_expiry_raises(
        self, mock_exchange: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = [0.0]
        monkeypatch.setattr("lmux_anthropic.auth._monotonic", lambda: clock[0])
        mock_exchange.side_effect = [
            ("sk-ant-oat01-1", 600.0),
            AuthenticationError("boom", provider="anthropic"),
        ]
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = provider.get_credentials()
        assert token_provider() == "sk-ant-oat01-1"
        clock[0] = 600.0  # the cached token has actually expired; the failure propagates
        with pytest.raises(AuthenticationError, match="boom"):
            token_provider()

    def test_first_use_exchange_failure_raises(self, mock_exchange: MagicMock) -> None:
        mock_exchange.side_effect = AuthenticationError("boom", provider="anthropic")
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        with pytest.raises(AuthenticationError, match="boom"):
            provider.get_credentials()()

    def test_identity_token_file_read_per_exchange(self, mock_exchange: MagicMock, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text("id-jwt-from-file\n")
        provider = _wif_provider(identity_token_file=token_file)
        provider.get_credentials()()
        mock_exchange.assert_called_once_with(
            assertion="id-jwt-from-file",
            federation_rule_id="fdrl_1",
            organization_id="org-1",
            service_account_id="svac_1",
            workspace_id=None,
            base_url=None,
        )

    def test_both_token_sources_raise(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _wif_provider(identity_token_provider=lambda: "id-jwt", identity_token_file="/token")  # noqa: S106

    def test_missing_token_source_raises(self) -> None:
        with pytest.raises(ValueError, match="identity_token_provider or identity_token_file"):
            _wif_provider()

    async def test_aget_credentials_returns_same_callable(self, mock_exchange: MagicMock) -> None:
        provider = _wif_provider(identity_token_provider=lambda: "id-jwt")
        token_provider = await provider.aget_credentials()
        assert token_provider == provider.get_credentials()  # the same bound resolver
        assert token_provider() == "sk-ant-oat01-1"
        mock_exchange.assert_called_once()


class TestAnthropicVertexADCAuthProvider:
    def test_get_credentials_falls_back_to_default(
        self,
        mock_google_auth_default: MagicMock,
        mock_credentials: MagicMock,
        mock_load_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        provider = AnthropicVertexADCAuthProvider()
        assert provider.get_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_load_credentials.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()

    def test_custom_scopes_for_default(
        self,
        mock_google_auth_default: MagicMock,
        mock_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        provider = AnthropicVertexADCAuthProvider(scopes=["https://www.googleapis.com/auth/custom"])
        assert provider.get_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/custom"])
        mock_cloud_sdk_path.assert_called_once_with()

    async def test_aget_credentials_falls_back_to_default(
        self,
        mock_google_auth_default: MagicMock,
        mock_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        provider = AnthropicVertexADCAuthProvider()
        assert await provider.aget_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_cloud_sdk_path.assert_called_once_with()

    def test_explicit_adc_file_uses_httpx_request(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_cloud_sdk_project: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (mock_credentials, "file-project")
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_with_scopes.assert_called_once_with(mock_credentials, [CLOUD_PLATFORM_SCOPE])
        mock_transport_request.assert_called_once_with()
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()
        mock_cloud_sdk_project.assert_not_called()

    def test_gcloud_adc_file_preserves_project_resolution(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_cloud_sdk_project: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "application_default_credentials.json"
        credentials_file.touch()
        mock_cloud_sdk_path.return_value = str(credentials_file)
        mock_load_credentials.return_value = (mock_credentials, None)

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (mock_credentials, "gcloud-project")
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_with_scopes.assert_called_once_with(mock_credentials, [CLOUD_PLATFORM_SCOPE])
        mock_cloud_sdk_project.assert_called_once_with()
        mock_google_auth_default.assert_not_called()
        mock_transport_request.assert_called_once_with()

    def test_external_account_project_uses_httpx_request(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_cloud_sdk_project: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "external.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
        scoped_credentials = MagicMock()
        scoped_credentials.get_project_id.return_value = "workload-project"
        mock_load_credentials.return_value = (mock_credentials, None)
        mock_with_scopes.return_value = scoped_credentials

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (scoped_credentials, "workload-project")
        scoped_credentials.get_project_id.assert_called_once_with(request=sentinel.request)
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()
        mock_cloud_sdk_project.assert_not_called()
        mock_transport_request.assert_called_once_with()

    def test_explicit_project_overrides_file_project(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_cloud_sdk_project: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "explicit-project")

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (mock_credentials, "explicit-project")
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_with_scopes.assert_called_once_with(mock_credentials, [CLOUD_PLATFORM_SCOPE])
        mock_credentials.get_project_id.assert_not_called()
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()
        mock_cloud_sdk_project.assert_not_called()
        mock_transport_request.assert_called_once_with()

    def test_file_without_project_lookup_returns_none(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_cloud_sdk_project: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
        credentials = object()
        mock_load_credentials.return_value = (credentials, None)
        mock_with_scopes.return_value = credentials

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (credentials, None)
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()
        mock_cloud_sdk_project.assert_not_called()
        mock_transport_request.assert_called_once_with()

    def test_missing_explicit_file_preserves_default_fallback(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        gcloud_file = tmp_path / "application_default_credentials.json"
        gcloud_file.touch()
        mock_cloud_sdk_path.return_value = str(gcloud_file)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))

        result = AnthropicVertexADCAuthProvider().get_credentials()

        assert result == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_load_credentials.assert_not_called()

    def test_custom_scopes_for_file_credentials(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_credentials: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_transport_request: MagicMock,
    ) -> None:
        custom_scopes = ["https://www.googleapis.com/auth/custom"]
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

        result = AnthropicVertexADCAuthProvider(scopes=custom_scopes).get_credentials()

        assert result == (mock_credentials, "file-project")
        mock_with_scopes.assert_called_once_with(mock_credentials, custom_scopes)
        mock_google_auth_default.assert_not_called()
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_cloud_sdk_path.assert_called_once_with()
        mock_transport_request.assert_called_once_with()

    def test_get_raises_import_error_without_extra(self, mock_missing_google_auth: None) -> None:
        assert mock_missing_google_auth is None  # side-effect fixture: patches sys.modules
        provider = AnthropicVertexADCAuthProvider()
        with pytest.raises(ImportError, match=r"\[vertex\] extra group is required"):
            _ = provider.get_credentials()


class TestAnthropicVertexServiceAccountAuthProvider:
    def test_get_credentials(self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        assert provider.get_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with("/path/to/key.json", scopes=[CLOUD_PLATFORM_SCOPE])

    def test_custom_scopes(self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(
            service_account_file="/path/to/key.json",
            scopes=["https://www.googleapis.com/auth/custom"],
        )
        assert provider.get_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/custom"]
        )

    async def test_aget_credentials(
        self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock
    ) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        assert await provider.aget_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with("/path/to/key.json", scopes=[CLOUD_PLATFORM_SCOPE])

    def test_get_raises_import_error_without_extra(self, mock_missing_google_oauth2: None) -> None:
        assert mock_missing_google_oauth2 is None  # side-effect fixture: patches sys.modules
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        with pytest.raises(ImportError, match=r"\[vertex\] extra group is required"):
            _ = provider.get_credentials()


class TestAnthropicFoundryEnvAuthProvider:
    def test_get_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "foundry-test-key")
        provider = AnthropicFoundryEnvAuthProvider()
        assert provider.get_credentials() == "foundry-test-key"

    def test_get_credentials_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        provider = AnthropicFoundryEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="ANTHROPIC_FOUNDRY_API_KEY") as exc_info:
            _ = provider.get_credentials()
        assert exc_info.value.provider == "anthropic-foundry"

    async def test_aget_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "foundry-test-key")
        provider = AnthropicFoundryEnvAuthProvider()
        assert await provider.aget_credentials() == "foundry-test-key"


class TestAnthropicFoundryTokenAuthProvider:
    def test_get_credentials_returns_token_provider(self) -> None:
        def token_provider() -> str:
            return "entra-token"  # pragma: no cover

        provider = AnthropicFoundryTokenAuthProvider(token_provider=token_provider)
        assert provider.get_credentials() is token_provider

    async def test_aget_credentials_returns_token_provider(self) -> None:
        def token_provider() -> str:
            return "entra-token"  # pragma: no cover

        provider = AnthropicFoundryTokenAuthProvider(token_provider=token_provider)
        assert await provider.aget_credentials() is token_provider


class TestAnthropicBedrockEnvAuthProvider:
    def test_get_credentials_returns_session(self) -> None:
        session = AnthropicBedrockEnvAuthProvider().get_credentials()
        assert isinstance(session, boto3.Session)

    async def test_aget_credentials_returns_session(self) -> None:
        session = await AnthropicBedrockEnvAuthProvider().aget_credentials()
        assert isinstance(session, boto3.Session)


class TestAnthropicBedrockSessionAuthProvider:
    def test_get_credentials_applies_config(self) -> None:
        provider = AnthropicBedrockSessionAuthProvider(region_name="us-west-2", profile_name=None)
        session = provider.get_credentials()
        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    async def test_aget_credentials_applies_config(self) -> None:
        provider = AnthropicBedrockSessionAuthProvider(region_name="eu-west-1")
        session = await provider.aget_credentials()
        assert session.region_name == "eu-west-1"
