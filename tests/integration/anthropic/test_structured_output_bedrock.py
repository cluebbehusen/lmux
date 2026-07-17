"""Native Anthropic-on-Bedrock structured output: a JSON-schema response format returns
content that parses and matches the schema, and cost matches the published rate.

Schema-constrained output rides on the native ``output_config`` field, which Bedrock's Converse
API does not expose — another capability the native transport exists to provide.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, JsonSchemaResponseFormat, UserMessage
from lmux_anthropic.auth import AnthropicBedrockSessionAuthProvider
from lmux_anthropic.provider import AnthropicBedrockProvider

_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_REGION = "us-east-1"
_MAX_TOKENS = 64
_PROMPT = "Reply with the word pong as the answer."
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

_RESPONSE_FORMAT = JsonSchemaResponseFormat(name="answer_only", json_schema=_SCHEMA)

_CASSETTE = Path(__file__).parent.parent / "cassettes" / "anthropic" / "bedrock_structured_output.json"

# claude-haiku-4-5 on Bedrock published rates ($/token, us-east-1).
_RATES = {"input_rate": 1.10 / 1_000_000, "output_rate": 5.50 / 1_000_000}


def _chat(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401 — harness-supplied per mode
    return AnthropicBedrockProvider(auth=auth, transport=transport, region=_REGION).chat(
        _MODEL, [UserMessage(content=_PROMPT)], max_tokens=_MAX_TOKENS, response_format=_RESPONSE_FORMAT
    )


@pytest.mark.verified
def test_structured_output_bedrock(
    scenario: Callable[..., Any],
    assert_chat: Callable[..., None],
    assert_cost: Callable[..., None],
    offline_bedrock_auth: AnthropicBedrockSessionAuthProvider,
) -> None:
    resp = scenario(_CASSETTE, _chat, requires="AWS_BEARER_TOKEN_BEDROCK", offline_auth=offline_bedrock_auth)
    assert_chat(resp, provider="anthropic-bedrock")
    payload = json.loads(resp.content)  # schema-constrained, so the content is parseable JSON
    assert "pong" in payload["answer"].lower()
    assert_cost(resp, **_RATES)
