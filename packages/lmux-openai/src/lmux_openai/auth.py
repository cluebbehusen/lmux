"""Authentication for OpenAI provider."""

import os

from lmux.exceptions import AuthenticationError


class OpenAIEnvAuthProvider:
    """Default auth provider that reads the API key from OPENAI_API_KEY env var."""

    def get_credentials(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            msg = "OPENAI_API_KEY environment variable is not set"
            raise AuthenticationError(msg, provider="openai")
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()
