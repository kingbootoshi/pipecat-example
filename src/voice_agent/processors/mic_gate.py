from __future__ import annotations

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    StartFrame,
    EndFrame,
    CancelFrame,
    StopFrame,
    InputAudioRawFrame,
)
import pipecat.frames.frames as _frames


class MicGate(FrameProcessor):
    """Hard gate for mic input/VAD frames.

    Drops mic audio and VAD/interruption frames whenever the provided
    `should_allow()` callable returns False. Always forwards lifecycle frames.
    """

    def __init__(self, should_allow_callable):
        super().__init__()
        self._should_allow = should_allow_callable

    async def process_frame(self, frame, direction: FrameDirection):
        # ensure base lifecycle hooks fire
        await super().process_frame(frame, direction)

        # always pass lifecycle frames
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame, StopFrame)):
            await self.push_frame(frame, direction)
            return

        # Build a compatible set of interruption/VAD frame classes across pipecat versions
        _maybe_types = [
            getattr(_frames, "StartInterruptionFrame", None),
            getattr(_frames, "StopInterruptionFrame", None),
            getattr(_frames, "UserStartedSpeakingFrame", None),
            getattr(_frames, "UserStoppedSpeakingFrame", None),
            getattr(_frames, "BotInterruptionFrame", None),
            getattr(_frames, "InterruptionFrame", None),
        ]
        _extra_types = tuple(t for t in _maybe_types if isinstance(t, type))

        # gate mic and VAD/interruption frames
        if isinstance(frame, (InputAudioRawFrame,) + _extra_types):
            if not bool(self._should_allow()):
                # drop
                return

        await self.push_frame(frame, direction)


__all__ = ["MicGate"]
