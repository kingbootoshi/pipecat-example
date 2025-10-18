#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def _resolve_conversation_file() -> Path:
    return (Path.cwd() / "conversations" / "conversation.json").expanduser().resolve()


def main() -> None:
    convo_path = _resolve_conversation_file()
    if convo_path.exists():
        convo_path.unlink()
        print(f"Deleted conversation history at {convo_path}")
    else:
        print(f"No conversation file to delete at {convo_path}")


if __name__ == "__main__":
    main()
