from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Iterable, List, Optional

from pydub import AudioSegment

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.local.audio import LocalAudioTransport

from ..constants import SFX_DIR
from ..logging import logger
from ..state import SharedState

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}


class SFXManager:
    """Loads and plays short sound effects through the local audio transport."""

    def __init__(
        self,
        *,
        state: SharedState,
        transport: LocalAudioTransport,
        directory: Path | None = None,
    ) -> None:
        self._state = state
        self._transport = transport
        base = Path(directory or SFX_DIR).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        self._dir = base
        self._dir.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        self._play_task: asyncio.Task | None = None
        self._playing_name: Optional[str] = None

    def list_tracks(self) -> List[str]:
        """Return available audio filenames in the SFX directory."""
        if not self._dir.exists():
            return []
        files: Iterable[Path] = (p for p in self._dir.iterdir() if p.is_file())
        filtered = [
            p.name
            for p in files
            if p.suffix.lower() in _AUDIO_EXTENSIONS and not p.name.startswith(".")
        ]
        return sorted(filtered, key=str.lower)

    def current_track(self) -> Optional[str]:
        return self._playing_name

    async def play(self, name: str) -> None:
        """Start playback of the named SFX in the background."""
        path = self._resolve(name)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)

        async with self._lock:
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._play_task

            task = asyncio.create_task(self._playback(path))
            self._play_task = task

    async def stop(self) -> None:
        """Stop any active playback."""
        async with self._lock:
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._play_task

    def _resolve(self, name: str) -> Path:
        target = (self._dir / name).expanduser()
        try:
            resolved = target.resolve(strict=False)
        except OSError:
            return target
        if self._dir not in resolved.parents and resolved != self._dir:
            # Protect against path traversal.
            raise FileNotFoundError(name)
        return resolved

    async def _playback(self, path: Path) -> None:
        task = asyncio.current_task()
        was_muted = self._state.force_muted
        was_speaking = self._state.tts_speaking
        self._state.set_force_muted(True)
        self._state.set_tts_speaking(True)

        self._playing_name = path.name
        try:
            segment = await asyncio.get_running_loop().run_in_executor(
                None, AudioSegment.from_file, path
            )

            output = self._transport.output()
            sample_rate = output.sample_rate or segment.frame_rate or 24000
            num_channels = 1
            normalized = (
                segment.set_frame_rate(sample_rate).set_channels(num_channels).set_sample_width(2)
            )
            raw = normalized.raw_data
            if not raw:
                logger.warning(f"[SFX] Empty audio data for {path.name}, skipping playback")
                return

            bytes_per_frame = num_channels * 2
            chunk_bytes = output.audio_chunk_size or max(
                int(sample_rate / 100) * bytes_per_frame, bytes_per_frame
            )

            for offset in range(0, len(raw), chunk_bytes):
                if task and task.cancelled():
                    break
                chunk = raw[offset : offset + chunk_bytes]
                if not chunk:
                    break

                frame = OutputAudioRawFrame(
                    audio=bytes(chunk),
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                )
                try:
                    await output.write_audio_frame(frame)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"[SFX] Failed writing audio chunk: {exc}")
                    break
        except asyncio.CancelledError:
            raise
        finally:
            self._playing_name = None
            if self._state.force_muted != was_muted:
                self._state.set_force_muted(was_muted)
            if not was_speaking:
                self._state.set_tts_speaking(False)
            if self._play_task is task:
                self._play_task = None
