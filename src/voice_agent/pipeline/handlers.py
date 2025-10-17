from __future__ import annotations

from pipecat.frames.frames import TTSStartedFrame, TTSStoppedFrame

from ..state import SharedState


def register_handlers(task, state: SharedState) -> None:
    """Attach minimal TTS start/stop listeners to update speaking state."""

    @task.event_handler("on_frame_reached_downstream")
    async def _tts_events(_processor, frame):  # noqa: D401
        if isinstance(frame, TTSStartedFrame):
            state.set_tts_speaking(True)
        elif isinstance(frame, TTSStoppedFrame):
            state.set_tts_speaking(False)


__all__ = ["register_handlers"]

