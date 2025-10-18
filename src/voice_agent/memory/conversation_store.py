from __future__ import annotations

import json
import os
from typing import Any

from ..constants import CONVERSATION_FILE, CONVERSATIONS_DIR, WEBCAM_MARKER
from ..logging import logger

MAX_LOADED_MESSAGES = 100
_ARCHIVED_MESSAGES: list[Any] = []


def save_conversation(context: Any) -> str:
    global _ARCHIVED_MESSAGES
    os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
    logger.info(f"Saving conversation to {CONVERSATION_FILE}")

    try:
        messages = getattr(context, "_messages", [])

        filtered_messages = []
        for msg in messages:
            # Drop any webcam marker messages entirely so we never persist
            # screenshot placeholders in history.
            if isinstance(msg.get("content"), list) and any(
                isinstance(item, dict)
                and item.get("type") == "text"
                and str(item.get("text", "")).startswith(WEBCAM_MARKER)
                for item in msg["content"]
            ):
                continue

            filtered_msg = msg.copy()
            if isinstance(msg.get("content"), list):
                filtered_content = [
                    item
                    for item in msg["content"]
                    if not (isinstance(item, dict) and item.get("type") == "image_url")
                ]

                if not filtered_content:
                    logger.debug("Skipping message with only image content")
                    continue

                filtered_msg["content"] = filtered_content

            filtered_messages.append(filtered_msg)

        full_history = [*_ARCHIVED_MESSAGES, *filtered_messages]

        with open(CONVERSATION_FILE, "w") as f:
            json.dump(full_history, f, indent=2)

        logger.info(
            f"Successfully saved conversation to {CONVERSATION_FILE} "
            f"({len(full_history)} messages persisted, images filtered out)"
        )

        if len(full_history) > MAX_LOADED_MESSAGES:
            _ARCHIVED_MESSAGES = full_history[:-MAX_LOADED_MESSAGES]
        else:
            _ARCHIVED_MESSAGES = []

        return CONVERSATION_FILE
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise


def load_conversation(context: Any) -> bool:
    global _ARCHIVED_MESSAGES
    if not os.path.exists(CONVERSATION_FILE):
        logger.info("No existing conversation file found. Starting fresh.")
        _ARCHIVED_MESSAGES = []
        return False

    logger.info(f"Loading conversation from {CONVERSATION_FILE}")

    try:
        with open(CONVERSATION_FILE, "r") as f:
            messages = json.load(f)
        total = len(messages) if isinstance(messages, list) else 0
        if total > MAX_LOADED_MESSAGES:
            _ARCHIVED_MESSAGES = messages[:-MAX_LOADED_MESSAGES]
            recent_messages = messages[-MAX_LOADED_MESSAGES:]
            logger.info(
                f"Loaded {MAX_LOADED_MESSAGES} most recent messages "
                f"(total history {total}, older messages kept on disk)"
            )
        else:
            _ARCHIVED_MESSAGES = []
            recent_messages = messages
            logger.info(f"Successfully loaded {total} messages from conversation history")

        setattr(context, "_messages", recent_messages)
        return True
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return False


__all__ = ["save_conversation", "load_conversation"]
