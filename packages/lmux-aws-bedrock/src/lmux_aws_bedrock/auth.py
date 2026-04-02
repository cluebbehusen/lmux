"""Authentication for AWS Bedrock provider.

The simplest way to authenticate is to set the ``AWS_BEARER_TOKEN_BEDROCK``
environment variable with a Bedrock API key and use ``BedrockEnvAuthProvider``
(the default). boto3/aiobotocore pick up this variable automatically via the
standard credential chain.

See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html

NOTE: boto3 does not yet support passing the bearer token as an explicit
session parameter. Once https://github.com/boto/boto3/issues/4723 is
addressed we should accept the key directly instead of relying on the env var.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import boto3
    from aiobotocore.session import AioSession


class BedrockEnvAuthProvider:
    """Default auth provider — creates bare sessions that inherit from the environment.

    Credentials are resolved by boto3's default credential chain:
    environment variables (``AWS_BEARER_TOKEN_BEDROCK``, ``AWS_ACCESS_KEY_ID``, …),
    the default profile, instance metadata, etc.
    """

    def get_credentials(self) -> "boto3.Session":
        import boto3  # noqa: PLC0415

        return boto3.Session()

    async def aget_credentials(self) -> "AioSession":
        try:
            from aiobotocore.session import get_session  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[async] extra group is required for async operations") from e  # noqa: TRY003

        return get_session()


class BedrockSessionAuthProvider:
    """Auth provider that creates sessions with explicit configuration.

    Accepts the same keyword arguments as ``boto3.Session`` and maps the
    async equivalents onto ``aiobotocore.session.get_session()``
    (``region_name``, ``profile_name``, ``aws_account_id``,
    ``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``).
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

    async def aget_credentials(self) -> "AioSession":
        try:
            from aiobotocore.session import get_session  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("[async] extra group is required for async operations") from e  # noqa: TRY003

        session = get_session()

        if self._kwargs["region_name"] is not None:
            session.set_config_variable("region", self._kwargs["region_name"])
        if self._kwargs["profile_name"] is not None:
            session.set_config_variable("profile", self._kwargs["profile_name"])
        if any(
            self._kwargs[key] is not None
            for key in (
                "aws_account_id",
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
            )
        ):
            session.set_credentials(
                self._kwargs["aws_access_key_id"],
                self._kwargs["aws_secret_access_key"],
                token=self._kwargs["aws_session_token"],
                account_id=self._kwargs["aws_account_id"],
            )

        return session
