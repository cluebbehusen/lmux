"""Vertex accepts both results from one parallel tool-call turn and uses them in its answer."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmux.types import ChatResponse, FunctionDefinition, Message, Tool, ToolMessage, UserMessage
from lmux_google import GoogleParams, GoogleProvider

_MODEL = "gemini-2.5-flash"
_MAX_TOKENS = 256
_NO_THINK = GoogleParams(thinking_config={"thinkingBudget": 0})
_CASSETTES = Path(__file__).parent.parent / "cassettes" / "google"
_CALL_CASSETTE = _CASSETTES / "vertex_parallel_tool_calls.json"
_RESULT_CASSETTE = _CASSETTES / "vertex_parallel_tool_results.json"
_REQUIRES = ("VERTEXAI_API_KEY", "GOOGLE_CLOUD_PROJECT")


@pytest.fixture
def code_tool() -> Tool:
    return Tool(
        function=FunctionDefinition(
            name="get_code",
            description="Look up the secret code for one label. Each label requires its own call.",
            parameters={
                "type": "object",
                "properties": {"label": {"type": "string", "enum": ["alpha", "beta"]}},
                "required": ["label"],
            },
        ),
    )


class TestParallelToolResponses:
    @pytest.mark.verified
    def test_parallel_tool_replay(
        self,
        scenario: Callable[..., Any],
        vertex_provider: Callable[..., GoogleProvider],
        assert_chat: Callable[..., None],
        code_tool: Tool,
    ) -> None:
        messages: list[Message] = [
            UserMessage(
                content="Call get_code for alpha and beta in parallel in the same turn, exactly once each. "
                "After receiving both results, repeat both codes verbatim in your answer."
            )
        ]

        def _first(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
            return vertex_provider(auth, transport).chat(
                _MODEL,
                messages,
                tools=[code_tool],
                tool_choice="required",
                max_tokens=_MAX_TOKENS,
                provider_params=_NO_THINK,
            )

        first = scenario(_CALL_CASSETTE, _first, requires=_REQUIRES)
        calls = first.tool_calls
        assert calls is not None, "The model must issue tool calls to exercise replay"
        assert len(calls) == 2, "The model must issue two calls in one turn to exercise replay"
        assert len({call.id for call in calls}) == 2
        assert [call.function.name for call in calls] == ["get_code", "get_code"]
        arguments = [json.loads(call.function.arguments) for call in calls]
        assert sorted(arguments, key=lambda args: args["label"]) == [{"label": "alpha"}, {"label": "beta"}]

        messages.append(first.to_assistant_message())
        codes = {"alpha": "ALPHA-731", "beta": "BETA-942"}
        for call, args in zip(calls, arguments, strict=True):
            messages.append(ToolMessage(tool_call_id=call.id, content=json.dumps({"code": codes[args["label"]]})))

        def _second(auth: Any, transport: Any) -> ChatResponse:  # noqa: ANN401
            return vertex_provider(auth, transport).chat(
                _MODEL,
                messages,
                tools=[code_tool],
                tool_choice="none",
                max_tokens=_MAX_TOKENS,
                provider_params=_NO_THINK,
            )

        final = scenario(_RESULT_CASSETTE, _second, requires=_REQUIRES)
        assert_chat(final, provider="google")
        assert not final.tool_calls
        assert final.finish_reason == "stop"
        assert final.content is not None
        assert all(code in final.content for code in codes.values()), "The answer must use both tool results"
