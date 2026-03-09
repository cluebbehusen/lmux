"""Tests for AWS Bedrock auth providers."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux_aws_bedrock.auth import BedrockEnvAuthProvider, BedrockSessionAuthProvider


@pytest.fixture
def mock_boto3_session_cls(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("boto3.Session")


@pytest.fixture
def mock_aioboto3_session_cls(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("aioboto3.Session")


@pytest.fixture
def mock_missing_aioboto3(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"aioboto3": None})


class TestBedrockEnvAuthProvider:
    def test_returns_boto3_session(self, mock_boto3_session_cls: MagicMock) -> None:
        provider = BedrockEnvAuthProvider()
        result = provider.get_credentials()

        assert result is mock_boto3_session_cls.return_value
        mock_boto3_session_cls.assert_called_once_with()

    async def test_aget_returns_aioboto3_session(self, mock_aioboto3_session_cls: MagicMock) -> None:
        provider = BedrockEnvAuthProvider()
        result = await provider.aget_credentials()

        assert result is mock_aioboto3_session_cls.return_value
        mock_aioboto3_session_cls.assert_called_once_with()

    async def test_aget_raises_import_error(self, mock_missing_aioboto3: None) -> None:
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

    async def test_aget_returns_aioboto3_session_with_kwargs(self, mock_aioboto3_session_cls: MagicMock) -> None:
        provider = BedrockSessionAuthProvider(
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret",  # noqa: S106
        )
        result = await provider.aget_credentials()

        assert result is mock_aioboto3_session_cls.return_value
        mock_aioboto3_session_cls.assert_called_once_with(
            region_name=None,
            profile_name=None,
            aws_account_id=None,
            aws_access_key_id="AKIA...",
            aws_secret_access_key="secret",  # noqa: S106
            aws_session_token=None,
        )

    async def test_aget_raises_import_error(self, mock_missing_aioboto3: None) -> None:
        provider = BedrockSessionAuthProvider()

        with pytest.raises(ImportError, match=r"\[async\] extra group is required for async operations.*"):
            await provider.aget_credentials()

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
