from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)


def make_local_audio_transport(
    audio_in_enabled: bool = True, audio_out_enabled: bool = True
) -> LocalAudioTransport:
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=audio_in_enabled,
            audio_out_enabled=audio_out_enabled,
        )
    )


__all__ = ["make_local_audio_transport"]

