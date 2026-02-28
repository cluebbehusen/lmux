"""Default environment-based auth provider for Groq."""

import os

from lmux.exceptions import AuthenticationError


class GroqEnvAuthProvider:
    """Auth provider that reads the API key from the GROQ_API_KEY environment variable."""

    def get_credentials(self) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None:
            msg = "GROQ_API_KEY environment variable is not set"
            raise AuthenticationError(msg, provider="groq")
        return api_key

    async def aget_credentials(self) -> str:
        return self.get_credentials()
