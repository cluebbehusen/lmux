"""Lazy AWS SDK loading internals.

Client creation is isolated here so tests can easily mock it
without patching sys.modules or using TYPE_CHECKING tricks.
"""

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import boto3
    from aiobotocore.session import AioSession
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient as AsyncBedrockRuntimeClient


def create_sync_client(
    session: "boto3.Session",
    *,
    region_name: str | None = None,
    endpoint_url: str | None = None,
) -> "BedrockRuntimeClient":
    """Create a sync bedrock-runtime client from a boto3.Session."""
    return session.client("bedrock-runtime", region_name=region_name, endpoint_url=endpoint_url)


def create_async_client(
    session: "AioSession",
    *,
    region_name: str | None = None,
    endpoint_url: str | None = None,
) -> AbstractAsyncContextManager["AsyncBedrockRuntimeClient"]:
    """Create an async bedrock-runtime client context manager from an aiobotocore session."""
    return session.create_client(
        "bedrock-runtime",
        region_name=region_name,
        endpoint_url=endpoint_url,
    )
