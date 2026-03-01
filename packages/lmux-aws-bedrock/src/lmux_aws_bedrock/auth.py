"""Authentication for AWS Bedrock provider.

The simplest way to authenticate is to set the ``AWS_BEARER_TOKEN_BEDROCK``
environment variable with a Bedrock API key and use ``BedrockEnvAuthProvider``
(the default). boto3/aioboto3 pick up this variable automatically via the
standard credential chain.

See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html

NOTE: boto3 does not yet support passing the bearer token as an explicit
session parameter. Once https://github.com/boto/boto3/issues/4723 is
addressed we should accept the key directly instead of relying on the env var.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aioboto3
    import boto3


class BedrockEnvAuthProvider:
    """Default auth provider — creates bare sessions that inherit from the environment.

    Credentials are resolved by boto3's default credential chain:
    environment variables (``AWS_BEARER_TOKEN_BEDROCK``, ``AWS_ACCESS_KEY_ID``, …),
    the default profile, instance metadata, etc.
    """

    def get_credentials(self) -> "boto3.Session":
        import boto3  # noqa: PLC0415

        return boto3.Session()

    async def aget_credentials(self) -> "aioboto3.Session":
        try:
            import aioboto3  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[async] extra group is required for async operations") from e  # noqa: TRY003

        return aioboto3.Session()


class BedrockSessionAuthProvider:
    """Auth provider that creates sessions with explicit configuration.

    Accepts the same keyword arguments as ``boto3.Session`` /
    ``aioboto3.Session`` (``region_name``, ``profile_name``,
    ``aws_account_id``, ``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``).
    Both sync and async sessions are constructed with the same kwargs.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_account_id: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None:
        self._kwargs: dict[str, Any] = {
            "region_name": region_name,
            "profile_name": profile_name,
            "aws_account_id": aws_account_id,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }

    def get_credentials(self) -> "boto3.Session":
        import boto3  # noqa: PLC0415

        return boto3.Session(**self._kwargs)

    async def aget_credentials(self) -> "aioboto3.Session":
        try:
            import aioboto3  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[async] extra group is required for async operations") from e  # noqa: TRY003

        return aioboto3.Session(**self._kwargs)
