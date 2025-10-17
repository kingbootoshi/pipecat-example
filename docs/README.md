# AI System Overview

This codebase uses a centralized AI module that routes all OpenAI‑compatible calls through OpenPipe to OpenRouter. We standardize on function (tool) calling only for consistent schemas and robust integrations.

- Client: `src/ai/client.py:1` (`AIClient`, `ChatMessage`, `build_standard_metadata`)
- Transport: OpenPipe SDK with `base_url=https://openrouter.ai/api/v1` (OpenRouter)
- Policy: Always use tool/function calling. Do not use structured outputs (`response_format`).
- Tests: `tests/test_ai_client.py:1` demonstrates end‑to‑end tool calling.

Quick start
- Copy `.env.example` to `.env` and fill keys.
- Run tests with uv: `uv run -m pytest -q tests/test_ai_client.py`

Key docs
- `docs/ai-module.md`: How to use the AI client.
- `docs/function-calling.md`: Standard patterns for tool/function calling.

