from __future__ import annotations

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..logging import logger
from ..memory.conversation_store import save_conversation


class ConversationAutosaveProcessor(FrameProcessor):
    """Persist conversation context whenever new turns reach the pipeline."""

    def __init__(self, context: LLMContext, *, label: str) -> None:
        super().__init__()
        self._context = context
        self._label = label
        self._last_saved_count = len(context.get_messages())

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and frame.context is self._context:
            message_count = len(self._context.get_messages())
            if message_count != self._last_saved_count:
                saved_path = save_conversation(self._context)
                logger.debug(
                    f"[autosave:{self._label}] conversation persisted to {saved_path} "
                    f"({message_count} messages)"
                )
                self._last_saved_count = message_count

        await self.push_frame(frame, direction)


__all__ = ["ConversationAutosaveProcessor"]
