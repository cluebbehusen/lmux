"""Offline auth for the Bedrock suite.

Bedrock's auth returns a boto3 session, not an API-key string, so the generic
harness stub does not fit. Offline replay still runs the provider's real request
build, which resolves credentials — a session carrying dummy keys lets SigV4 sign
locally (the replay transport serves the cassette, so the signature is never sent).
"""

import pytest

from lmux_aws_bedrock.auth import BedrockSessionAuthProvider

REGION = "us-east-1"


@pytest.fixture
def offline_auth() -> BedrockSessionAuthProvider:
    return BedrockSessionAuthProvider(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 — dummy key; the signature is never sent offline
        region_name=REGION,
    )
