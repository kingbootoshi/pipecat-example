from __future__ import annotations

import json
import os
from typing import Any

from ..constants import CONVERSATION_FILE, CONVERSATIONS_DIR
from ..logging import logger


def save_conversation(context: Any) -> str:
    os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
    logger.info(f"Saving conversation to {CONVERSATION_FILE}")

    try:
        messages = getattr(context, "_messages", [])

        filtered_messages = []
        for msg in messages:
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

        with open(CONVERSATION_FILE, "w") as f:
            json.dump(filtered_messages, f, indent=2)

        logger.info(
            f"Successfully saved conversation to {CONVERSATION_FILE} "
            f"({len(filtered_messages)} messages, images filtered out)"
        )
        return CONVERSATION_FILE
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise


def load_conversation(context: Any) -> bool:
    if not os.path.exists(CONVERSATION_FILE):
        logger.info("No existing conversation file found. Starting fresh.")
        return False

    logger.info(f"Loading conversation from {CONVERSATION_FILE}")

    try:
        with open(CONVERSATION_FILE, "r") as f:
            messages = json.load(f)
            setattr(context, "_messages", messages)
        logger.info(f"Successfully loaded {len(messages)} messages from conversation history")
        return True
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return False


__all__ = ["save_conversation", "load_conversation"]

