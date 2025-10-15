#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import glob
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# Single persistent conversation file
CONVERSATION_FILE = "./conversations/conversation.json"


class TranscriptionLogger(FrameProcessor):
    """Logs transcribed speech input from the user."""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"🎤 User said: {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)


class LLMResponseLogger(FrameProcessor):
    """Logs complete LLM responses from OpenRouter by aggregating streaming tokens."""
    
    def __init__(self):
        super().__init__()
        self.response_buffer = ""  # Buffer to collect streaming tokens
        self.is_streaming = False  # Track if we're currently streaming a response
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # Collect streaming tokens in buffer
            self.response_buffer += frame.text
            self.is_streaming = True
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            # End of response - log the complete response
            if self.is_streaming and self.response_buffer.strip():
                print(f"🤖 AI Response: {self.response_buffer.strip()}")
            
            # Reset buffer and streaming state
            self.response_buffer = ""
            self.is_streaming = False

        # Push all frames through
        await self.push_frame(frame, direction)


# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def save_conversation(context: LLMContext) -> str:
    """
    Save the conversation history to the persistent JSON file.
    
    Args:
        context: The LLMContext containing conversation messages
        
    Returns:
        The filename where conversation was saved
    """
    # Ensure the conversations directory exists
    os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
    
    logger.info(f"Saving conversation to {CONVERSATION_FILE}")
    
    try:
        with open(CONVERSATION_FILE, "w") as file:
            # LLMContext stores messages in '_messages' private attribute
            messages = context._messages
            json.dump(messages, file, indent=2)
        logger.info(f"Successfully saved conversation to {CONVERSATION_FILE}")
        return CONVERSATION_FILE
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise


def load_conversation(context: LLMContext) -> bool:
    """
    Load the conversation history from the persistent JSON file.
    
    Args:
        context: The LLMContext to load messages into
        
    Returns:
        True if conversation was loaded, False if file doesn't exist
    """
    if not os.path.exists(CONVERSATION_FILE):
        logger.info("No existing conversation file found. Starting fresh.")
        return False
    
    logger.info(f"Loading conversation from {CONVERSATION_FILE}")
    
    try:
        with open(CONVERSATION_FILE, "r") as file:
            messages = json.load(file)
            # LLMContext stores messages in '_messages' private attribute
            context._messages = messages
        logger.info(f"Successfully loaded {len(messages)} messages from conversation history")
        return True
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return False


async def run_bot(transport: LocalAudioTransport):
    """
    Run the bot with local audio transport for speech input, transcription, and AI responses.
    Automatically loads and saves conversation history.
    
    Args:
        transport: The LocalAudioTransport instance
    """
    logger.info("Starting conversational AI bot with local audio transport")

    # Initialize AssemblyAI STT service for speech-to-text
    stt = AssemblyAISTTService(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
    )

    # Initialize OpenRouter LLM service for AI responses
    llm = OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meta-llama/llama-3.3-70b-instruct",
    )

    # Create LLM context with system message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Your responses will be spoken aloud, so keep them concise and conversational. Respond naturally to what the user says.",
        },
    ]
    context = LLMContext(messages)

    # Always try to load existing conversation history
    load_conversation(context)

    # Create context aggregator to manage conversation state
    context_aggregator = LLMContextAggregatorPair(context)

    # Initialize processors for logging
    transcription_logger = TranscriptionLogger()
    llm_response_logger = LLMResponseLogger()

    # Create pipeline with proper context management
    pipeline = Pipeline([
        transport.input(),              # Receives audio from microphone
        stt,                            # Converts audio to text
        transcription_logger,           # Logs the transcribed text
        context_aggregator.user(),      # Aggregates user context for LLM
        llm,                            # Processes text and generates AI response
        llm_response_logger,            # Logs the AI response
        context_aggregator.assistant(), # Aggregates assistant responses
    ])

    # Create pipeline task
    task = PipelineTask(pipeline)

    # Create pipeline runner with signal handling
    runner = PipelineRunner(handle_sigint=True)

    try:
        # Run the pipeline
        await runner.run(task)
    finally:
        # Save conversation history on exit
        try:
            saved_file = save_conversation(context)
            logger.info(f"Conversation saved to {saved_file}")
        except Exception as e:
            logger.error(f"Failed to save conversation on exit: {e}")


async def main():
    """Main entry point for conversational AI bot with local audio input."""
    # Check for required API keys
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        logger.error("ASSEMBLYAI_API_KEY environment variable is required")
        return
    
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable is required")
        return

    # Check if persistent conversation file exists
    if os.path.exists(CONVERSATION_FILE):
        logger.info(f"Existing conversation found at {CONVERSATION_FILE}")
        logger.info("Conversation history will be loaded automatically")
    else:
        logger.info("No existing conversation. Starting fresh.")

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