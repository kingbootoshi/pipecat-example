#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
import sys
from datetime import datetime

import cv2
import yaml
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
# Import STT mute filter to prevent processing user input while bot is speaking
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat_whisker import WhiskerObserver

load_dotenv(override=True)

# Single persistent conversation file
CONVERSATION_FILE = "./conversations/conversation.json"
# Configuration file for system prompt and other settings
CONFIG_FILE = "./config.yml"


def load_config() -> dict:
    """
    Load configuration from config.yml file.
    
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config.yml doesn't exist
        yaml.YAMLError: If config.yml is not valid YAML
    """
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Configuration file not found: {CONFIG_FILE}")
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    try:
        with open(CONFIG_FILE, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {CONFIG_FILE}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse configuration file: {e}")
        raise


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


class ImageCaptureProcessor(FrameProcessor):
    """
    Captures webcam image after each transcription and adds it to the LLM context.
    Ensures only the most recent image exists in context by removing old ones before adding new ones.
    This allows the LLM to see the user's webcam with each interaction.
    
    Note: Images are only kept in the runtime context and are NOT saved to conversation.json
    (filtered out during save to keep file size manageable).
    """
    
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self._webcam_marker = "__webcam__"  # Marker to identify webcam messages
        self._webcam = None  # Webcam capture object
        self._init_webcam()
    
    def _init_webcam(self):
        """
        Initialize the webcam capture object.
        Uses camera index 0 (default/built-in camera).
        """
        try:
            logger.info("📹 Initializing webcam...")
            self._webcam = cv2.VideoCapture(0)
            
            if not self._webcam.isOpened():
                logger.error("❌ Could not open webcam")
                self._webcam = None
            else:
                # Get camera properties
                width = int(self._webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"✅ Webcam initialized: {width}x{height}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize webcam: {e}")
            self._webcam = None
    
    def _take_webcam_photo(self) -> tuple[bytes, tuple[int, int], str] | None:
        """
        Captures a photo from the webcam.
        
        Returns:
            Tuple of (raw_image_bytes, (width, height), image_mode) or None if capture fails
            
        Note: Returns RAW pixel bytes (not encoded), image mode (e.g. "RGB"),
        as required by Pipecat's add_image_frame_message()
        """
        if self._webcam is None or not self._webcam.isOpened():
            logger.warning("⚠️ Webcam not available, skipping webcam capture")
            return None
        
        logger.debug("Starting webcam capture...")
        
        try:
            # Capture frame
            ret, frame = self._webcam.read()
            
            if not ret:
                logger.error("❌ Failed to capture webcam frame")
                return None
            
            logger.debug(f"Webcam frame captured: shape={frame.shape}")
            
            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            logger.debug(f"Webcam image: mode={img.mode}, size={img.size}")
            
            # Return RAW pixel bytes (not encoded) - Pipecat will encode it
            raw_bytes = img.tobytes()
            logger.debug(f"Extracted webcam raw pixel bytes: {len(raw_bytes)} bytes")
            
            # Return raw bytes, size, and PIL image mode
            return raw_bytes, img.size, img.mode
            
        except Exception as e:
            logger.error(f"❌ Error capturing webcam photo: {e}")
            return None
    
    def _remove_old_images(self):
        """
        Removes any existing webcam messages from the context.
        This ensures we only keep the most recent image.
        """
        # Access the internal messages list
        if hasattr(self._context, "_messages"):
            # Filter out messages that contain our webcam marker
            self._context._messages = [
                msg for msg in self._context._messages
                if not (isinstance(msg.get("content"), list) and 
                       any(item.get("text", "").startswith(self._webcam_marker)
                           for item in msg.get("content", []) if isinstance(item, dict)))
            ]
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # When we receive a transcription, capture webcam image
        if isinstance(frame, TranscriptionFrame):
            logger.info("🎥 Capturing webcam for LLM context")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Take a webcam photo (returns raw bytes, size, and PIL image mode, or None if unavailable)
                webcam_result = self._take_webcam_photo()
                if not webcam_result:
                    logger.warning("⚠️ Webcam capture failed, skipping image context")
                    await self.push_frame(frame, direction)
                    return
                
                webcam_bytes, webcam_size, webcam_mode = webcam_result
                logger.debug(f"Webcam captured: bytes_len={len(webcam_bytes)}, size={webcam_size}, mode={webcam_mode}")
                
                # Save webcam image locally for debugging
                webcam_dir = "./screenshots"
                os.makedirs(webcam_dir, exist_ok=True)
                
                try:
                    webcam_path = os.path.join(webcam_dir, f"webcam_{timestamp}.jpg")
                    debug_webcam = Image.frombytes(webcam_mode, webcam_size, webcam_bytes)
                    debug_webcam.save(webcam_path)
                    logger.debug(f"💾 Webcam saved to: {webcam_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")
                
                # Remove any old webcam images from context
                logger.debug("Removing old webcam images from context...")
                self._remove_old_images()
                logger.debug("Old images removed")
                
                # Log context state before adding
                logger.debug(f"Context has {len(self._context._messages)} messages before adding webcam")
                
                # Add the webcam image to context
                logger.debug(f"Adding webcam image: size={webcam_size}, mode={webcam_mode}")
                self._context.add_image_frame_message(
                    image=webcam_bytes,
                    text=f"{self._webcam_marker} User webcam view",
                    size=webcam_size,
                    format=webcam_mode,
                )
                
                logger.debug(f"Context now has {len(self._context._messages)} messages after adding webcam")
                logger.info(f"✅ Webcam image added to context: {webcam_size}")
                
            except Exception as e:
                logger.error(f"❌ Failed to capture webcam: {e}")
                logger.exception("Full traceback:")
        
        # Always push frames through to maintain pipeline flow
        await self.push_frame(frame, direction)


# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def save_conversation(context: LLMContext) -> str:
    """
    Save the conversation history to the persistent JSON file.
    Filters out image data (image_url type) to keep the file size manageable.
    
    Args:
        context: The LLMContext containing conversation messages
        
    Returns:
        The filename where conversation was saved
    """
    # Ensure the conversations directory exists
    os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
    
    logger.info(f"Saving conversation to {CONVERSATION_FILE}")
    
    try:
        # LLMContext stores messages in '_messages' private attribute
        messages = context._messages
        
        # Filter out image data from messages to reduce file size
        filtered_messages = []
        for msg in messages:
            filtered_msg = msg.copy()
            
            # Check if content is a list (multimodal content)
            if isinstance(msg.get("content"), list):
                # Filter out image_url type content items
                filtered_content = [
                    item for item in msg["content"]
                    if not (isinstance(item, dict) and item.get("type") == "image_url")
                ]
                
                # If all content was filtered out, skip this message entirely
                if not filtered_content:
                    logger.debug(f"Skipping message with only image content")
                    continue
                    
                filtered_msg["content"] = filtered_content
            
            filtered_messages.append(filtered_msg)
        
        with open(CONVERSATION_FILE, "w") as file:
            json.dump(filtered_messages, file, indent=2)
        
        logger.info(f"Successfully saved conversation to {CONVERSATION_FILE} ({len(filtered_messages)} messages, images filtered out)")
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

    # Load configuration first
    config = load_config()
    
    # Initialize AssemblyAI STT service for speech-to-text
    stt = AssemblyAISTTService(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
    )

    # Initialize OpenRouter LLM service for AI responses with config values
    llm_config = config.get("llm", {})
    llm = OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=llm_config.get("model", "anthropic/claude-haiku-4.5"),
    )

    # Initialize ElevenLabs TTS service for audio output with config values
    elevenlabs_config = config.get("elevenlabs", {})
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=elevenlabs_config.get("voice_id", "mNeKLtUk8yWjz7uDi1dj"),
        stability=elevenlabs_config.get("stability", 0.5),
        similarity_boost=elevenlabs_config.get("similarity_boost", 0.75),
    )

    # Get system prompt from config
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    # Create LLM context with system message from config
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]
    context = LLMContext(messages)

    # Always try to load existing conversation history
    load_conversation(context)

    # Create context aggregator to manage conversation state
    context_aggregator = LLMContextAggregatorPair(context)

    # Initialize processors for logging and image capture
    transcription_logger = TranscriptionLogger()
    llm_response_logger = LLMResponseLogger()
    image_capture_processor = ImageCaptureProcessor(context)  # Captures webcam, adds to context
    
    # 🔇 Configure STT mute filter to prevent processing user input while bot is speaking
    # Using ALWAYS strategy to mute user input during all bot speech instances
    stt_mute_processor = STTMuteFilter(
        config=STTMuteConfig(
            strategies={STTMuteStrategy.ALWAYS}
        ),
    )

    # Create pipeline with image capture before context aggregation
    pipeline = Pipeline([
        transport.input(),              # Receives audio from microphone
        stt,                            # Converts audio to text
        stt_mute_processor,             # 🔇 Mute user input while bot is speaking
        transcription_logger,           # Logs the transcribed text
        image_capture_processor,        # Captures webcam, adds to context (before aggregation)
        context_aggregator.user(),      # Aggregates user context for LLM
        llm,                            # Processes text and generates AI response
        tts,                            # Converts LLM text to speech
        transport.output(),             # Sends audio to speakers
        context_aggregator.assistant(), # Aggregates assistant responses (after output)
        llm_response_logger,            # Logs the AI response (after aggregation)
    ])

    # Create Whisker observer for pipeline monitoring
    whisker = WhiskerObserver(pipeline)

    # Create pipeline params with interruptions disabled
    # This prevents user speech from interrupting the bot while it's speaking
    params = PipelineParams(
        allow_interruptions=False,
    )

    # Create pipeline task with Whisker observer and params
    task = PipelineTask(pipeline, params=params, observers=[whisker])

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
    """Main entry point for conversational AI bot with local audio input and output."""
    # Check for required API keys
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        logger.error("ASSEMBLYAI_API_KEY environment variable is required")
        return
    
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable is required")
        return
    
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.error("ELEVENLABS_API_KEY environment variable is required")
        return

    # Check if persistent conversation file exists
    if os.path.exists(CONVERSATION_FILE):
        logger.info(f"Existing conversation found at {CONVERSATION_FILE}")
        logger.info("Conversation history will be loaded automatically")
    else:
        logger.info("No existing conversation. Starting fresh.")

    # Create local audio transport with input and output enabled
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,                                        # Enable microphone input
            audio_out_enabled=True,                                       # Enable speaker output
        )
    )

    # Run the bot with the local audio transport
    await run_bot(transport)


if __name__ == "__main__":
    asyncio.run(main())