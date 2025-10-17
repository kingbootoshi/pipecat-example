from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext

from .config import load_config
from .logging import setup_logging, logger
from .memory.conversation_store import load_conversation, save_conversation
from .observers.whisker import make_whisker_observer
from .pipeline.builder import build_pipeline
from .pipeline.params import make_params
from .services.llm import make_llm
from .services.stt import make_stt
from .services.tts import make_tts
from .settings import load_env, missing_required_keys
from .transports.local_audio import make_local_audio_transport


class VoiceAgent:
    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        vision: bool = True,
        memory: bool = True,
        allow_interruptions: bool = False,
        observers: Optional[Iterable[Any]] = None,
        transport: Optional[Any] = None,
        log_level: str = "DEBUG",
    ) -> None:
        setup_logging(log_level)
        load_env()
        missing = missing_required_keys()
        if missing:
            logger.error(
                "Missing required environment variables: " + ", ".join(missing)
            )
            raise RuntimeError("Missing required environment variables")

        self._config = load_config(config_path)

        self._system_prompt = self._config.get(
            "system_prompt", "You are a helpful assistant."
        )

        # Build context with system prompt
        self._context = LLMContext(
            [
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            ]
        )

        if memory:
            load_conversation(self._context)

        # Services
        self._stt = make_stt()
        self._llm = make_llm(self._config.get("llm", {}))
        self._tts = make_tts(self._config.get("elevenlabs", {}))

        # Transport
        self._transport = transport or make_local_audio_transport(
            audio_in_enabled=True, audio_out_enabled=True
        )

        # Pipeline
        self._pipeline, self._aggregator, _ = build_pipeline(
            transport=self._transport,
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            context=self._context,
            enable_vision=vision,
        )

        # Observers (defer Whisker creation until start())
        self._observers = list(observers or [])

        # Params and base task (without Whisker)
        self._params = make_params(allow_interruptions=allow_interruptions)
        self._task = PipelineTask(
            self._pipeline, params=self._params, observers=self._observers
        )
        # Runner will be created inside start() when an event loop is running
        self._runner: Optional[PipelineRunner] = None

        self._memory_enabled = memory
        self._run_task: Optional[asyncio.Task] = None

    def build_task(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> tuple[PipelineTask, Optional[PipelineRunner]]:
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; caller can create the runner later
                return self._task, None
        runner = PipelineRunner(loop=loop, handle_sigint=True)
        return self._task, runner

    def get_context(self) -> LLMContext:
        return self._context

    async def start(self) -> None:
        logger.info("Starting VoiceAgent")
        try:
            # Create a task with dynamic observers (attach Whisker now that loop is running)
            dynamic_observers = list(self._observers)
            w = make_whisker_observer(self._pipeline)
            if w:
                dynamic_observers.append(w)
            task = PipelineTask(
                self._pipeline, params=self._params, observers=dynamic_observers
            )
            # Create runner bound to the current running loop
            self._runner = PipelineRunner(handle_sigint=True)
            await self._runner.run(task)
        finally:
            if self._memory_enabled:
                try:
                    saved_file = save_conversation(self._context)
                    logger.info(f"Conversation saved to {saved_file}")
                except Exception as e:
                    logger.error(f"Failed to save conversation on exit: {e}")

    def start_background(self) -> None:
        if self._run_task and not self._run_task.done():
            return
        loop = asyncio.get_event_loop()
        self._run_task = loop.create_task(self.start())

    def stop(self) -> None:
        # Best-effort stop. Depending on pipecat internals, this may vary.
        try:
            if hasattr(self._task, "stop"):
                self._task.stop()  # type: ignore[attr-defined]
        except Exception:
            pass


__all__ = ["VoiceAgent"]
