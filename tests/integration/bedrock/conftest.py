"""Offline auth for the Bedrock suite.

Bedrock's auth returns a boto3 session, not an API-key string, so the generic harness
stub does not fit. This stub returns fake SigV4 credentials directly, without constructing
a boto3 session, so offline replay never reads ambient AWS configuration: a stale
AWS_PROFILE (or ~/.aws) would otherwise make boto3 raise ProfileNotFound before the replay
transport is reached. The provider still signs the request; the signature is served to the
replay transport and never sent.
"""

from typing import TYPE_CHECKING, cast

import pytest

from lmux_aws_bedrock.auth import BedrockSessionAuthProvider

if TYPE_CHECKING:
    import boto3
    from aiobotocore.session import AioSession

REGION = "us-east-1"


class _FrozenCreds:
    access_key = "testing"
    secret_key = "testing"  # noqa: S105 — dummy key; the signature is never sent offline
    token = None


class _Credentials:
    def get_frozen_credentials(self) -> "_FrozenCreds":
        return _FrozenCreds()


class _Session:
    region_name = REGION

    def get_credentials(self) -> "_Credentials":
        return _Credentials()


class _OfflineBedrockAuth(BedrockSessionAuthProvider):
    """Resolve fake credentials without a boto3 session, keeping offline replay hermetic."""

    def get_credentials(self) -> "boto3.Session":
        return cast("boto3.Session", _Session())

    async def aget_credentials(self) -> "AioSession":
        return cast("AioSession", _Session())


@pytest.fixture
def offline_auth() -> BedrockSessionAuthProvider:
    return _OfflineBedrockAuth()
