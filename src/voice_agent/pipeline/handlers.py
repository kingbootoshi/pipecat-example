from __future__ import annotations

import asyncio
import time
from loguru import logger
from pipecat.frames.frames import (
    TTSStartedFrame,
    TTSStoppedFrame,
    OutputAudioRawFrame,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)

from ..state import SharedState


def register_handlers(task, state: SharedState) -> None:
    """Attach minimal TTS start/stop listeners to update speaking state."""

    # Restrict which frames invoke our handlers for performance and to ensure
    # OutputAudioRawFrame events are delivered (some builds require explicit filters).
    try:
        from pipecat.frames.frames import (
            LLMMessagesFrame as _U,
            LLMMessagesAppendFrame as _UA,
            LLMFullResponseStartFrame as _RS,
            LLMFullResponseEndFrame as _RE,
            LLMTextFrame as _RT,
            TTSStartedFrame as _TS,
            TTSStoppedFrame as _TE,
            OutputAudioRawFrame as _OA,
        )
        # Optionally include transport-level bot speaking frames if available
        try:
            from pipecat.frames.frames import (
                BotStartedSpeakingFrame as _BS,
                BotStoppedSpeakingFrame as _BE,
            )
            _bot = (_BS, _BE)
        except Exception:
            _bot = tuple()

        task.set_reached_upstream_filter((_U, _UA))
        task.set_reached_downstream_filter((_UA, _RS, _RE, _RT, _TS, _TE, _OA) + _bot)
        logger.debug("[handlers] filters set for upstream/downstream frame taps (incl Bot* if present)")
    except Exception as exc:
        logger.debug(f"[handlers] filter setup skipped: {exc}")

    @task.event_handler("on_frame_reached_downstream")
    async def _tts_events(_processor, frame):  # noqa: D401
        if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
            logger.debug("[speech] started (TTS/BotStarted)")
            state.set_tts_speaking(True)
        elif isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            logger.debug("[speech] stopped (TTS/BotStopped)")
            state.set_tts_speaking(False)

    # Fallback: infer speaking from OutputAudioRawFrame presence across builds
    last_audio = {"ts": 0.0}
    watchdog = {"task": None}

    @task.event_handler("on_frame_reached_downstream")
    async def _audio_seen(_processor, frame):  # noqa: D401
        if isinstance(frame, OutputAudioRawFrame):
            last_audio["ts"] = time.monotonic()
            if not state.tts_speaking:
                logger.debug("[speech] audio frame → speaking ON")
                state.set_tts_speaking(True)
            # start watchdog lazily when loop is guaranteed to be running
            if watchdog["task"] is None:
                try:
                    watchdog["task"] = asyncio.create_task(_silence_watchdog())
                except Exception as exc:
                    logger.debug(f"[speech] watchdog start failed: {exc}")

    async def _silence_watchdog():
        idle_after = 0.35  # seconds of no audio → not speaking
        while True:
            try:
                await asyncio.sleep(0.1)
                if state.tts_speaking and (time.monotonic() - last_audio["ts"]) > idle_after:
                    logger.debug("[speech] silence timeout → speaking OFF")
                    state.set_tts_speaking(False)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.25)



__all__ = ["register_handlers"]
