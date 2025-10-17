# Function (Tool) Calling Standard

We exclusively use function/tool calling to integrate models with application logic. This ensures consistent schemas and predictable flows.

Contract
- Requests: `chat.completions.create` with `tools=[{ type: "function", function: { ... } }]`
- Responses: parse `choices[0].message.tool_calls` and execute each call
- Tool output messages: append as `{ role: "tool", tool_call_id: <id>, content: <string> }`
- Follow‑up: send the enriched `messages` back to the model to get the final response

Schema rules (strict mode)
- Include `"strict": true` in the function definition when supported
- `parameters.additionalProperties: false`
- Every property in `parameters.properties` that the tool needs must appear in `parameters.required`
- Optional properties should use union types with `null` (e.g. `"type": ["string", "null"]`)

Minimal template
```python
tools = [{
  "type": "function",
  "function": {
    "name": "<name>",
    "description": "<what it does>",
    "parameters": {
      "type": "object",
      "properties": { ... },
      "required": [ ... ],
      "additionalProperties": False
    },
    "strict": True,
  }
}]

resp1 = client.chat(
  messages=messages,
  model="openai/gpt-4o-mini",
  tools=tools,
  tool_choice="required", # or "auto"
  temperature=0,
)

assistant_msg = resp1.choices[0].message
tool_calls = assistant_msg.tool_calls

# Append the assistant message with tool_calls before tool outputs
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
    output = my_router(tc.function.name, args)
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(output),
    })

resp2 = client.chat(
  messages=messages,
  model="openai/gpt-4o-mini",
  tools=tools,
  temperature=0,
)
final_text = resp2.choices[0].message.content
```

Tool design guidelines
- Clear names and descriptions; state when to use the tool
- Use enums and types to prevent invalid states
- Keep tool count small per request (< 20) for accuracy
- Avoid asking the model to provide data your code already holds
- Combine steps that are always executed together

Parallel tool calling
- If the model could call multiple tools, allow `parallel_tool_calls=True`
- If exactly zero or one call is required, set `parallel_tool_calls=False`

Allowed tools / tool choice
- `tool_choice="auto"`: model decides
- `tool_choice="required"`: force one or more calls
- `tool_choice={"type":"function","name":"<name>"}`: force exactly one specific function

Streaming (optional)
- Enable with `stream=True`, then accumulate `chunk.choices[0].delta.tool_calls`
- Only use streaming when needed; non‑streaming is simpler and is used in tests

Anti‑patterns
- Do not use `response_format` structured outputs in this repo
- Do not mix direct free‑text outputs with tool outputs in a single step; always return tool outputs first, then ask for final text

References
- Client: `src/ai/client.py:1`
- Example test: `tests/test_ai_client.py:1`
