# AI Module

Centralized access to LLMs via OpenPipe (OpenAI‑compatible SDK) routed to OpenRouter. This module is decoupled from the voice pipeline and is the only approved way to invoke LLMs from this repo.

Files
- `src/ai/client.py:1`
  - `AIClient`: main entry point
  - `ChatMessage`: lightweight message container
  - `build_standard_metadata`: helper to enforce metadata conventions
- `tests/test_ai_client.py:1`: reference usage and tool‑calling flow

Environment
- Required
  - `OPENROUTER_API_KEY`: OpenRouter key
  - `OPENPIPE_API_KEY`: OpenPipe key
- Optional
  - `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
  - `OPENPIPE_BASE_URL`
  - `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME` (OpenRouter attribution headers)
  - `OPENROUTER_MODEL` (default `openai/gpt-4o-mini` in tests)

Instantiation
```python
from ai import AIClient, ChatMessage

client = AIClient.from_env()
```

Policy
- Always use tool/function calling (chat.completions with `tools`).
- Do not use `response_format` structured outputs in this codebase.
- Prefer explicit, strict schemas (`additionalProperties: false`, full `required`).

Standard metadata
```python
from ai.client import build_standard_metadata

md = build_standard_metadata(
    app="voice_agent",
    component="<component>",
    function="<function>",
    conversation_id="<convo-id>",
    turn_index=1,
    tags=["example"],
)
```
Fields:
- `app`, `component`, `function`: required identifiers
- `request_id`: auto‑generated UUID if omitted
- `conversation_id`, `turn_index`, `tags`, plus any extra keys you pass

Function calling example
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

messages = [
    {"role": "system", "content": "You are precise."},
    {"role": "user", "content": "Please add a=2 and b=3 using the tool."},
]

# 1) Request with tools, require a tool call
resp1 = client.chat(
    messages=messages,
    model="openai/gpt-4o-mini",
    metadata=md,
    temperature=0,
    tools=tools,
    tool_choice="required",
)

# 2) Append assistant tool-call message, then execute tools and append results
assistant_msg = resp1.choices[0].message
tool_calls = assistant_msg.tool_calls  # attribute access in SDK

# Important: The tool result messages must immediately follow the assistant
# message that contains `tool_calls`.
messages.append({
    "role": "assistant",
    "content": assistant_msg.content,
    "tool_calls": [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ],
})

for tc in tool_calls:
    args = json.loads(tc.function.arguments)
    result = {"sum": int(args["a"]) + int(args["b"])}
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(result),
    })

# 3) Ask model to produce final text
resp2 = client.chat(
    messages=messages,
    model="openai/gpt-4o-mini",
    metadata=md,
    temperature=0,
    tools=tools,
)
final_text = resp2.choices[0].message.content
```

Notes
- `AIClient.chat(...)` accepts `tools`, `tool_choice`, `parallel_tool_calls`, `stream`, and forwards them to the OpenAI‑compatible API.
- Structured outputs via `response_format` are not used in this repo. Use function/tool calling only.
