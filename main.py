#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)


# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(transport: LocalAudioTransport):
    """Run the bot with local audio transport for speech input and transcription."""
    logger.info("Starting bot with local audio transport")

    # Initialize AssemblyAI STT service
    stt = AssemblyAISTTService(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
    )

    # Initialize transcription logger to display transcribed text
    transcription_logger = TranscriptionLogger()

    # Create pipeline: audio input -> STT -> transcription logging
    pipeline = Pipeline([
        transport.input(),      # Receives audio from microphone
        stt,                    # Converts audio to text
        transcription_logger,    # Logs the transcribed text
    ])

    # Create pipeline task
    task = PipelineTask(pipeline)

    # Create pipeline runner with signal handling
    runner = PipelineRunner(handle_sigint=True)

    # Run the pipeline
    await runner.run(task)


async def main():
    """Main entry point for local audio transcription bot."""
    # Create local audio transport with input enabled
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,   # Enable microphone input
            audio_out_enabled=False, # Disable audio output for now
        )
    )

    # Run the bot with the local audio transport
    await run_bot(transport)


if __name__ == "__main__":
    asyncio.run(main())