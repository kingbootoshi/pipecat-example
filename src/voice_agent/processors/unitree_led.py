from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, Tuple

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import StartFrame, EndFrame, CancelFrame, StopFrame

from ..state import SharedState
from ..logging import logger

# --- Static imports for Unitree SDK (no dynamic imports at runtime) ---------
HAS_UNITREE = True
IMPORT_ERR: Exception | None = None
try:  # import-time detection only
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize as _CFI
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient as _AudioClient
except Exception as _exc:  # pragma: no cover - environment dependent
    HAS_UNITREE = False
    IMPORT_ERR = _exc


Color = Tuple[int, int, int]


class _UnitreeLEDClient:
    """Wrapper around Unitree AudioClient LED control, with simulation fallback.

    Attempts to initialize DDS + AudioClient; if anything fails, falls back to
    simulation mode where LED set calls are logged but not sent.
    """

    def __init__(self, iface: str | None) -> None:
        self._iface = iface or os.getenv("UNITREE_INTERFACE")
        self._sim = True
        self._client = None

    def init(self) -> None:
        if not HAS_UNITREE:
            logger.debug(f"[UnitreeLED] SDK not available, simulating: {IMPORT_ERR}")
            self._sim = True
            return
        try:
            # DDS/channel init. Will raise on failure per SDK behavior
            _CFI(0, self._iface)
            c = _AudioClient()
            c.SetTimeout(5.0)
            c.Init()
            self._client = c
            self._sim = False
            try:
                modpath = getattr(_AudioClient, "__module__", "unitree_sdk2py")
            except Exception:
                modpath = "unitree_sdk2py"
            logger.info(f"[UnitreeLED] Connected via interface '{self._iface}' ({modpath})")
        except Exception as exc:
            logger.warning(f"[UnitreeLED] Init failed, simulating: {exc}")
            self._sim = True
            self._client = None

    def set_color(self, r: int, g: int, b: int) -> None:
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        if self._sim or self._client is None:
            logger.debug(f"[UnitreeLED][SIM] set RGB ({r},{g},{b})")
            return
        try:
            self._client.LedControl(r, g, b)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(f"[UnitreeLED] LedControl error, switching to sim: {exc}")
            self._sim = True

    def is_sim(self) -> bool:
        return self._sim


class UnitreeLEDProcessor(FrameProcessor):
    """Controls Unitree LED based on speaking/listening state.

    Policy:
    - speaking (tts_speaking=True): flash on/off at a cadence.
    - listening (True) and not speaking: solid light pink.
    - else (muted and not speaking): LED off.

    When SDK/DDS init fails (common on mac during dev), we simulate and log.
    """

    def __init__(
        self,
        state: Optional[SharedState],
        *,
        iface: Optional[str] = None,
        pink: Color = (255, 10, 10),  # Default pink per hardware (user-defined)
        blink_hz: float = 2.0,
    ) -> None:
        super().__init__()
        self.state = state
        # Primary blink color (pink) and a dimmed variant for visible pink blink
        self._pink = pink
        self._off: Color = (0, 0, 0)
        # Allow tuning via env; default faster blink (2x) if not provided
        try:
            blink_env = float(os.getenv("UNITREE_LED_HZ", "4.0"))
            blink_hz = blink_env if blink_env > 0 else blink_hz
        except Exception:
            pass
        self._blink_period = 1.0 / max(0.1, float(blink_hz))
        try:
            # Default to a very low (near-off) dim level per request
            dim_env = float(os.getenv("UNITREE_LED_DIM", "0.15"))
            dim_factor = min(1.0, max(0.0, dim_env))
        except Exception:
            dim_factor = 0.15
        self._dim_pink: Color = (
            max(0, min(255, int(self._pink[0] * dim_factor))),
            max(0, min(255, int(self._pink[1] * dim_factor))),
            max(0, min(255, int(self._pink[2] * dim_factor))),
        )
        self._tick = 0.1
        self._led = _UnitreeLEDClient(iface)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_set: Optional[Color] = None
        self._blink_on = False
        self._blink_t0 = time.monotonic()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Best-effort init (may simulate if fails)
            self._led.init()
            # Immediately set baseline pink at startup regardless of listening state
            self._apply(self._pink)
            # Start loop if not already
            if self._task is None:
                self._stop.clear()
                self._task = asyncio.create_task(self._loop())

        elif isinstance(frame, (EndFrame, CancelFrame, StopFrame)):
            self._stop.set()
            if self._task:
                try:
                    await self._task
                except Exception:
                    pass
                self._task = None

        await self.push_frame(frame, direction)

    def _compute_target(self) -> Color:
        s = self.state
        now = time.monotonic()
        if s and s.tts_speaking:
            # Blink between bright pink and dim pink (avoid hardware defaulting to white)
            if (now - self._blink_t0) >= (self._blink_period / 2.0):
                self._blink_on = not self._blink_on
                self._blink_t0 = now
            return self._pink if self._blink_on else self._dim_pink
        # Baseline: always pink when not speaking (requested behavior)
        return self._pink

    async def _loop(self) -> None:
        # initial set
        target = self._compute_target()
        self._apply(target)

        while not self._stop.is_set():
            try:
                target = self._compute_target()
                # Only apply when color actually changes
                if target != self._last_set:
                    self._apply(target)
            except Exception as exc:
                logger.debug(f"[UnitreeLED] loop error: {exc}")
            await asyncio.sleep(self._tick)

    def _apply(self, color: Color) -> None:
        r, g, b = color
        # Log real hardware sets as well as sim for visibility
        logger.debug(f"[UnitreeLED] apply RGB ({r},{g},{b})")
        self._led.set_color(r, g, b)
        self._last_set = color


__all__ = ["UnitreeLEDProcessor"]
