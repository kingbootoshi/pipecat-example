from __future__ import annotations

from typing import Any


def make_whisker_observer(pipeline: Any):
    try:
        from pipecat_whisker import WhiskerObserver
    except Exception:
        # Whisker is optional; return None if not available
        return None

    return WhiskerObserver(pipeline)


__all__ = ["make_whisker_observer"]

