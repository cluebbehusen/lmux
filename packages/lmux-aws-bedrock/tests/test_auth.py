"""Tests for AWS Bedrock auth providers."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux_aws_bedrock.auth import BedrockEnvAuthProvider, BedrockSessionAuthProvider


@pytest.fixture
def mock_boto3_session_cls(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("boto3.Session")


@pytest.fixture
def mock_aiobotocore_get_session(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("aiobotocore.session.get_session")


@pytest.fixture
def mock_missing_aiobotocore(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"aiobotocore.session": None})


class TestBedrockEnvAuthProvider:
    def test_returns_boto3_session(self, mock_boto3_session_cls: MagicMock) -> None:
        provider = BedrockEnvAuthProvider()
        result = provider.get_credentials()

        assert result is mock_boto3_session_cls.return_value
        mock_boto3_session_cls.assert_called_once_with()

    async def test_aget_returns_aiobotocore_session(self, mock_aiobotocore_get_session: MagicMock) -> None:
        provider = BedrockEnvAuthProvider()
        result = await provider.aget_credentials()

        assert result is mock_aiobotocore_get_session.return_value
        mock_aiobotocore_get_session.assert_called_once_with()

    async def test_aget_raises_import_error(self, mock_missing_aiobotocore: None) -> None:
        provider = BedrockEnvAuthProvider()

        with pytest.raises(ImportError, match=r"\[async\] extra group is required for async operations.*"):
            await provider.aget_credentials()


class TestBedrockSessionAuthProvider:
    def test_returns_boto3_session_with_kwargs(self, mock_boto3_session_cls: MagicMock) -> None:
        provider = BedrockSessionAuthProvider(
            region_name="us-west-2",
            profile_name="my-profile",
        )
        result = provider.get_credentials()

        assert result is mock_boto3_session_cls.return_value
        mock_boto3_session_cls.assert_called_once_with(
            region_name="us-west-2",
            profile_name="my-profile",
            aws_account_id=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        )

    async def test_aget_returns_aiobotocore_session_with_kwargs(self, mock_aiobotocore_get_session: MagicMock) -> None:
        provider = BedrockSessionAuthProvider(
            region_name="us-west-2",
            profile_name="my-profile",
            aws_account_id="123456789012",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret",  # noqa: S106
            aws_session_token="session-token",  # noqa: S106
        )
        result = await provider.aget_credentials()

        assert result is mock_aiobotocore_get_session.return_value
        mock_aiobotocore_get_session.assert_called_once_with()
        mock_aiobotocore_get_session.return_value.set_config_variable.assert_any_call("region", "us-west-2")
        mock_aiobotocore_get_session.return_value.set_config_variable.assert_any_call("profile", "my-profile")
        mock_aiobotocore_get_session.return_value.set_credentials.assert_called_once_with(
            "AKIA...",
            "secret",
            token="session-token",  # noqa: S106
            account_id="123456789012",
        )

    async def test_aget_raises_import_error(self, mock_missing_aiobotocore: None) -> None:
        provider = BedrockSessionAuthProvider()

        with pytest.raises(ImportError, match=r"\[async\] extra group is required for async operations.*"):
            await provider.aget_credentials()

    async def test_aget_with_default_kwargs_does_not_set_config_or_credentials(
        self, mock_aiobotocore_get_session: MagicMock
    ) -> None:
        provider = BedrockSessionAuthProvider()

        result = await provider.aget_credentials()

        assert result is mock_aiobotocore_get_session.return_value
        mock_aiobotocore_get_session.assert_called_once_with()
        mock_aiobotocore_get_session.return_value.set_config_variable.assert_not_called()
        mock_aiobotocore_get_session.return_value.set_credentials.assert_not_called()

    async def test_aget_account_id_only_does_not_override_credential_chain(
        self, mock_aiobotocore_get_session: MagicMock
    ) -> None:
        provider = BedrockSessionAuthProvider(aws_account_id="123456789012")

        await provider.aget_credentials()

        mock_aiobotocore_get_session.return_value.set_credentials.assert_not_called()

    def test_default_kwargs_all_none(self, mock_boto3_session_cls: MagicMock) -> None:
        provider = BedrockSessionAuthProvider()
        provider.get_credentials()

        mock_boto3_session_cls.assert_called_once_with(
            region_name=None,
            profile_name=None,
            aws_account_id=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        )
