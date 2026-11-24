"""Unit tests for LiteLLM tool_choice: the caller's tool_choice must reach litellm instead of a hard-coded "auto"."""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("litellm")

from agno.models.litellm import LiteLLM
from agno.models.message import Message

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "", "parameters": {}}}]
NAMED_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


def _model() -> LiteLLM:
    model = LiteLLM(id="openai/gpt-4o-mini", api_key="test-key")
    model._parse_provider_response = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    model._parse_provider_response_delta = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return model


def _messages():
    return [Message(role="user", content="hi")]


def test_request_params_default_tool_choice_is_auto():
    params = LiteLLM(id="openai/gpt-4o-mini", api_key="test-key").get_request_params(tools=TOOLS)
    assert params["tool_choice"] == "auto"


@pytest.mark.parametrize("tool_choice", ["none", "required", NAMED_CHOICE])
def test_request_params_forward_tool_choice(tool_choice):
    params = LiteLLM(id="openai/gpt-4o-mini", api_key="test-key").get_request_params(
        tools=TOOLS, tool_choice=tool_choice
    )
    assert params["tool_choice"] == tool_choice


def test_invoke_forwards_tool_choice():
    model = _model()
    model.client = MagicMock()

    model.invoke(_messages(), Message(role="assistant"), tools=TOOLS, tool_choice="required")

    assert model.client.completion.call_args.kwargs["tool_choice"] == "required"


def test_invoke_stream_forwards_tool_choice():
    model = _model()
    model.client = MagicMock()
    model.client.completion.return_value = iter([])

    list(model.invoke_stream(_messages(), Message(role="assistant"), tools=TOOLS, tool_choice=NAMED_CHOICE))

    assert model.client.completion.call_args.kwargs["tool_choice"] == NAMED_CHOICE


@pytest.mark.asyncio
async def test_ainvoke_forwards_tool_choice():
    model = _model()
    model.client = MagicMock()
    model.client.acompletion = AsyncMock()

    await model.ainvoke(_messages(), Message(role="assistant"), tools=TOOLS, tool_choice="none")

    assert model.client.acompletion.call_args.kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_ainvoke_stream_forwards_tool_choice():
    async def empty_stream():
        return
        yield

    model = _model()
    model.client = MagicMock()
    model.client.acompletion = AsyncMock(return_value=empty_stream())

    async for _ in model.ainvoke_stream(_messages(), Message(role="assistant"), tools=TOOLS, tool_choice="required"):
        pass

    assert model.client.acompletion.call_args.kwargs["tool_choice"] == "required"
