import os
from typing import Mapping, Any

from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


def make_tts(elevenlabs_config: Mapping[str, Any] | None = None) -> ElevenLabsTTSService:
    cfg = dict(elevenlabs_config or {})
    return ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=cfg.get("voice_id", "mNeKLtUk8yWjz7uDi1dj"),
        stability=cfg.get("stability", 0.5),
        similarity_boost=cfg.get("similarity_boost", 0.75),
    )


__all__ = ["make_tts"]

