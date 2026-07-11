"""lmux-aws-bedrock — AWS Bedrock provider for lmux."""

import contextlib

from lmux_aws_bedrock.auth import BedrockEnvAuthProvider, BedrockSessionAuthProvider
from lmux_aws_bedrock.cost import calculate_bedrock_cost
from lmux_aws_bedrock.params import BedrockParams, GuardrailConfig
from lmux_aws_bedrock.provider import BedrockProvider

__all__ = [
    "BedrockEnvAuthProvider",
    "BedrockParams",
    "BedrockProvider",
    "BedrockSessionAuthProvider",
    "GuardrailConfig",
    "calculate_bedrock_cost",
    "preload",
]


def preload() -> None:
    """Eagerly import httpx and boto3 (and aiobotocore if installed).

    httpx is the request transport; boto3 resolves AWS credentials for SigV4 signing.
    Call this during application startup to pay the import cost upfront rather than on
    the first request.
    """
    import boto3  # noqa: F401, PLC0415
    import httpx  # noqa: F401, PLC0415

    with contextlib.suppress(ImportError):
        import aiobotocore  # noqa: F401, PLC0415
