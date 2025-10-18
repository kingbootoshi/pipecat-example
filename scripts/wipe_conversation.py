#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_conversation_file() -> Path:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from voice_agent.constants import CONVERSATION_FILE  # noqa: WPS433

    return root / CONVERSATION_FILE


def main() -> None:
    convo_path = _resolve_conversation_file()
    if convo_path.exists():
        convo_path.unlink()
        print(f"Deleted conversation history at {convo_path}")
    else:
        print(f"No conversation file to delete at {convo_path}")


if __name__ == "__main__":
    main()
