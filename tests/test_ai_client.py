import json
import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("OPENPIPE_API_KEY"),
    reason="OPENROUTER_API_KEY and OPENPIPE_API_KEY must be set to run this test.",
)


def _attr_or_dict(obj, attr, key):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return obj[key]


def test_function_calling_add_numbers():
    # Lazy import to avoid ImportError when deps are missing
    from ai.client import AIClient, build_standard_metadata

    client = AIClient.from_env()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers and return their sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First integer"},
                        "b": {"type": "integer", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": "Please add a=2 and b=3 using the add_numbers tool."},
    ]

    md = build_standard_metadata(
        app="voice_agent",
        component="ai_module_test",
        function="test_function_calling_add_numbers",
        conversation_id="test-convo",
        turn_index=1,
        tags=["test", "function-calling"],
    )

    # 1) Ask the model and require it to call a tool
    resp1 = client.chat(
        messages=messages,
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        metadata=md,
        temperature=0,
        tools=tools,
        tool_choice="required",
    )

    # 2) Extract tool calls and simulate executing them
    if hasattr(resp1, "choices"):
        choice0 = resp1.choices[0]
        message = _attr_or_dict(choice0, "message", "message")
        tool_calls = _attr_or_dict(message, "tool_calls", "tool_calls")
    else:
        tool_calls = resp1["choices"][0]["message"].get("tool_calls", [])

    assert tool_calls, "Model did not return any tool calls"

    # Append the assistant tool-call message before tool outputs
    if hasattr(resp1, "choices"):
        # Convert SDK object to a plain dict structure acceptable by the API
        assistant_msg = {
            "role": "assistant",
            "content": _attr_or_dict(message, "content", "content") if hasattr(message, "content") or (isinstance(message, dict) and "content" in message) else None,
            "tool_calls": [
                {
                    "id": _attr_or_dict(tc, "id", "id"),
                    "type": "function",
                    "function": {
                        "name": _attr_or_dict(_attr_or_dict(tc, "function", "function"), "name", "name"),
                        "arguments": _attr_or_dict(_attr_or_dict(tc, "function", "function"), "arguments", "arguments"),
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)
    else:
        messages.append(resp1["choices"][0]["message"])  # already dict

    for tc in tool_calls:
        # OpenAI SDK returns objects with attributes; fallback to dict index when needed
        func = _attr_or_dict(tc, "function", "function")
        name = _attr_or_dict(func, "name", "name")
        args_json = _attr_or_dict(func, "arguments", "arguments")
        call_id = _attr_or_dict(tc, "id", "id")

        assert name == "add_numbers"
        args = json.loads(args_json)
        total = int(args["a"]) + int(args["b"])  # simulate tool execution

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"sum": total, "a": args["a"], "b": args["b"]}),
            }
        )

    # 3) Send tool outputs back to the model and get final response
    resp2 = client.chat(
        messages=messages,
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        metadata=md,
        temperature=0,
        tools=tools,
    )

    if hasattr(resp2, "choices"):
        choice0 = resp2.choices[0]
        message = _attr_or_dict(choice0, "message", "message")
        content = _attr_or_dict(message, "content", "content")
    else:
        content = resp2["choices"][0]["message"]["content"]

    assert isinstance(content, str)
    assert content.strip() != ""
