from __future__ import annotations

from typing import Any, Iterable

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from ..processors.transcription_logger import TranscriptionLogger
from ..processors.llm_response_logger import LLMResponseLogger
from ..processors.image_capture import ImageCaptureProcessor
from ..processors.filters import make_stt_mute_filter_always
from ..processors.mic_gate import MicGate
from ..state import SharedState


def build_pipeline(
    *,
    transport,
    stt,
    llm,
    tts,
    context: LLMContext,
    enable_vision: bool = True,
) -> tuple[Pipeline, LLMContextAggregatorPair, dict[str, Any]]:
    context_aggregator = LLMContextAggregatorPair(context)

    transcription_logger = TranscriptionLogger()
    llm_response_logger = LLMResponseLogger()
    stt_mute_processor = make_stt_mute_filter_always()

    # Resolve global/shared state for mic gating; allow when listening and not speaking
    # Caller should import and set a SharedState on the VoiceAgent. We lazily
    # import here to avoid circulars in app initialization.
    # If no SharedState is available at runtime, default to allowing mic.
    try:
        from ..state import SharedState as _S  # type: ignore
        # find a module-level singleton if set by the app (assigned later)
        state: SharedState | None = getattr(_S, "_singleton", None)  # type: ignore[attr-defined]
    except Exception:
        state = None

    def _should_allow() -> bool:
        if isinstance(state, SharedState):
            return bool(state.listening) and not bool(state.tts_speaking)
        return True

    mic_gate = MicGate(_should_allow)

    processors_pre_agg: list[Any] = [
        transport.input(),
        mic_gate,
        stt,
        stt_mute_processor,
        transcription_logger,
    ]

    if enable_vision:
        processors_pre_agg.append(ImageCaptureProcessor(context))

    pipeline = Pipeline(
        processors_pre_agg
        + [
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
            llm_response_logger,
        ]
    )

    return pipeline, context_aggregator, {
        "transcription_logger": transcription_logger,
        "llm_response_logger": llm_response_logger,
        "stt_mute_processor": stt_mute_processor,
        "mic_gate": mic_gate,
    }


__all__ = ["build_pipeline"]
