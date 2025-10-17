from __future__ import annotations

import asyncio
from typing import Optional
import contextlib

from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from .app import VoiceAgent
from .controller.server import create_api, run_http_server
from .observers.whisker import make_whisker_observer
from .pipeline.handlers import register_handlers


async def serve(agent: VoiceAgent, host: str = "127.0.0.1", port: int = 8710) -> None:
    """Run the HTTP server and pipeline concurrently in one loop."""
    # Build a dynamic task with optional Whisker observer
    dynamic_observers = list(agent._observers)  # type: ignore[attr-defined]
    w = make_whisker_observer(agent._pipeline)  # type: ignore[attr-defined]
    if w:
        dynamic_observers.append(w)
    task = PipelineTask(agent._pipeline, params=agent._params, observers=dynamic_observers)  # type: ignore[attr-defined]
    register_handlers(task, agent.get_state())

    runner = PipelineRunner(handle_sigint=True)

    api = create_api(agent.get_state())

    server_task = asyncio.create_task(run_http_server(api, host=host, port=port))
    pipeline_task = asyncio.create_task(runner.run(task))

    try:
        await asyncio.gather(server_task, pipeline_task)
    finally:
        for t in (server_task, pipeline_task):
            if not t.done():
                t.cancel()
                with contextlib.suppress(Exception):
                    await t
