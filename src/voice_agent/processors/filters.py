from pipecat.processors.filters.stt_mute_filter import (
    STTMuteConfig,
    STTMuteFilter,
    STTMuteStrategy,
)


def make_stt_mute_filter_always() -> STTMuteFilter:
    return STTMuteFilter(
        config=STTMuteConfig(strategies={STTMuteStrategy.ALWAYS}),
    )


__all__ = ["make_stt_mute_filter_always"]

