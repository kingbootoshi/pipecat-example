from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List


StateListener = Callable[[str, Any], None]


@dataclass
class SharedState:
    """Mutable app state with simple change notifications.

    Tracks whether the mic should be listening (`listening`) and whether the
    TTS is currently speaking (`tts_speaking`).
    """

    listening: bool = False
    tts_speaking: bool = False

    _listeners: List[StateListener] = field(default_factory=list)

    def add_listener(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def _notify(self, event: str, value: Any) -> None:
        for fn in list(self._listeners):
            try:
                fn(event, value)
            except Exception:
                # best-effort notifications
                pass

    def set_listening(self, value: bool) -> None:
        value = bool(value)
        if value != self.listening:
            self.listening = value
            self._notify("listening_changed", self.listening)

    def set_tts_speaking(self, value: bool) -> None:
        value = bool(value)
        if value != self.tts_speaking:
            self.tts_speaking = value
            self._notify("tts_speaking_changed", self.tts_speaking)


__all__ = ["SharedState", "StateListener"]

