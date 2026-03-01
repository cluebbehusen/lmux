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
    """Eagerly import boto3 (and aioboto3 if installed).

    Call this during application startup to pay the import cost upfront
    rather than on the first request.
    """
    import boto3  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]

    with contextlib.suppress(ImportError):
        import aioboto3  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport]
