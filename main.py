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
from io import BytesIO

import cv2
import mss
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat_whisker import WhiskerObserver

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


class ImageCaptureProcessor(FrameProcessor):
    """
    Captures both a screenshot and webcam image after each transcription and adds them to the LLM context.
    Ensures only the most recent images exist in context by removing old ones before adding new ones.
    This allows the LLM to see both the screen state and the user's webcam with each interaction.
    """
    
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self._screenshot_marker = "__screenshot__"  # Marker to identify screenshot messages
        self._webcam_marker = "__webcam__"  # Marker to identify webcam messages
        self._webcam = None  # Webcam capture object
        self._init_webcam()
    
    def _take_screenshot(self) -> tuple[bytes, tuple[int, int], str]:
        """
        Captures a screenshot of the primary monitor.
        
        Returns:
            Tuple of (raw_image_bytes, (width, height), image_mode)
            
        Note: Returns RAW pixel bytes (not encoded), image mode (e.g. "RGB"),
        as required by Pipecat's add_image_frame_message()
        """
        logger.debug("Starting screenshot capture...")
        
        with mss.mss() as sct:
            # Capture the primary monitor
            logger.debug(f"Available monitors: {sct.monitors}")
            monitor = sct.monitors[1]
            logger.debug(f"Capturing monitor: {monitor}")
            
            screenshot = sct.grab(monitor)
            logger.debug(f"Screenshot captured: size={screenshot.size}, width={screenshot.width}, height={screenshot.height}")
            
            # Log available screenshot attributes to debug
            logger.debug(f"Screenshot has bgra attr: {hasattr(screenshot, 'bgra')}")
            logger.debug(f"Screenshot has rgb attr: {hasattr(screenshot, 'rgb')}")
            logger.debug(f"Screenshot has rgba attr: {hasattr(screenshot, 'rgba')}")
            
            try:
                # Convert to PIL Image (BGRA format on macOS/Windows)
                logger.debug("Attempting BGRA conversion...")
                img = Image.frombytes("RGBA", screenshot.size, screenshot.bgra, "raw", "BGRA")
                logger.debug(f"BGRA conversion successful! Image mode: {img.mode}, size: {img.size}")
                
            except Exception as e:
                logger.error(f"BGRA conversion failed: {e}")
                logger.debug("Trying RGB conversion as fallback...")
                try:
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    logger.debug(f"RGB conversion successful! Image mode: {img.mode}, size: {img.size}")
                except Exception as e2:
                    logger.error(f"RGB conversion also failed: {e2}")
                    raise
            
            # Convert to RGB for better compatibility with LLMs
            logger.debug(f"Converting to RGB... current mode: {img.mode}")
            img = img.convert("RGB")
            logger.debug(f"Converted to RGB mode: {img.mode}")
            
            # Return RAW pixel bytes (not encoded) - Pipecat will encode it
            raw_bytes = img.tobytes()
            logger.debug(f"Extracted raw pixel bytes: {len(raw_bytes)} bytes")
            
            # Return raw bytes, size, and PIL image mode (not file format!)
            return raw_bytes, img.size, img.mode
    
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
        Removes any existing screenshot and webcam messages from the context.
        This ensures we only keep the most recent images.
        """
        # Access the internal messages list
        if hasattr(self._context, "_messages"):
            # Filter out messages that contain our image markers
            self._context._messages = [
                msg for msg in self._context._messages
                if not (isinstance(msg.get("content"), list) and 
                       any(item.get("text", "").startswith((self._screenshot_marker, self._webcam_marker))
                           for item in msg.get("content", []) if isinstance(item, dict)))
            ]
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # When we receive a transcription, capture both screenshot and webcam images
        if isinstance(frame, TranscriptionFrame):
            logger.info("📸🎥 Capturing screen and webcam for LLM context")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Take the screenshot (returns raw bytes, size, and PIL image mode)
                screenshot_bytes, screenshot_size, screenshot_mode = self._take_screenshot()
                logger.debug(f"Screenshot captured: bytes_len={len(screenshot_bytes)}, size={screenshot_size}, mode={screenshot_mode}")
                
                # Take a webcam photo (returns raw bytes, size, and PIL image mode, or None if unavailable)
                webcam_result = self._take_webcam_photo()
                if webcam_result:
                    webcam_bytes, webcam_size, webcam_mode = webcam_result
                    logger.debug(f"Webcam captured: bytes_len={len(webcam_bytes)}, size={webcam_size}, mode={webcam_mode}")
                else:
                    logger.warning("⚠️ Webcam capture skipped")
                
                # Save images locally for debugging
                screenshot_dir = "./screenshots"
                os.makedirs(screenshot_dir, exist_ok=True)
                
                try:
                    # Save screenshot
                    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
                    debug_img = Image.frombytes(screenshot_mode, screenshot_size, screenshot_bytes)
                    debug_img.save(screenshot_path)
                    logger.debug(f"💾 Screenshot saved to: {screenshot_path}")
                    
                    # Save webcam image if available
                    if webcam_result:
                        webcam_path = os.path.join(screenshot_dir, f"webcam_{timestamp}.jpg")
                        debug_webcam = Image.frombytes(webcam_mode, webcam_size, webcam_bytes)
                        debug_webcam.save(webcam_path)
                        logger.debug(f"💾 Webcam saved to: {webcam_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug images: {e}")
                
                # Remove any old images from context
                logger.debug("Removing old images from context...")
                self._remove_old_images()
                logger.debug("Old images removed")
                
                # Log context state before adding
                logger.debug(f"Context has {len(self._context._messages)} messages before adding images")
                
                # Add the screenshot to context
                logger.debug(f"Adding screenshot: size={screenshot_size}, mode={screenshot_mode}")
                self._context.add_image_frame_message(
                    image=screenshot_bytes,
                    text=f"{self._screenshot_marker} Current screen view",
                    size=screenshot_size,
                    format=screenshot_mode,
                )
                
                # Add the webcam image to context if available
                if webcam_result:
                    logger.debug(f"Adding webcam image: size={webcam_size}, mode={webcam_mode}")
                    self._context.add_image_frame_message(
                        image=webcam_bytes,
                        text=f"{self._webcam_marker} User webcam view",
                        size=webcam_size,
                        format=webcam_mode,
                    )
                
                logger.debug(f"Context now has {len(self._context._messages)} messages after adding images")
                logger.info(f"✅ Images added to context (screenshot: {screenshot_size}, webcam: {webcam_size if webcam_result else 'N/A'})")
                
            except Exception as e:
                logger.error(f"❌ Failed to capture images: {e}")
                logger.exception("Full traceback:")
        
        # Always push frames through to maintain pipeline flow
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
        model="anthropic/claude-haiku-4.5",
    )

    # Initialize ElevenLabs TTS service for audio output
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="FM4U7dtcVAjQuuSwraJ0",  # Specified voice ID
    )

    # Create LLM context with system message that includes vision capability
    messages = [
        {
            "role": "system",
            "content": "Your name is JARVIS. Your current developer is BOOTOSHI the blasian with curly hair",
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
    image_capture_processor = ImageCaptureProcessor(context)  # Captures screenshot + webcam, adds to context

    # Create pipeline with image capture before context aggregation
    pipeline = Pipeline([
        transport.input(),              # Receives audio from microphone
        stt,                            # Converts audio to text
        transcription_logger,           # Logs the transcribed text
        image_capture_processor,        # Captures screen + webcam, adds to context (before aggregation)
        context_aggregator.user(),      # Aggregates user context for LLM
        llm,                            # Processes text and generates AI response
        tts,                            # Converts LLM text to speech
        transport.output(),             # Sends audio to speakers
        context_aggregator.assistant(), # Aggregates assistant responses (after output)
        llm_response_logger,            # Logs the AI response (after aggregation)
    ])

    # Create Whisker observer for pipeline monitoring
    whisker = WhiskerObserver(pipeline)

    # Create pipeline task with Whisker observer
    task = PipelineTask(pipeline, observers=[whisker])

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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),  # Voice activity detection
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),  # Interruptible turn detection
        )
    )

    # Run the bot with the local audio transport
    await run_bot(transport)


if __name__ == "__main__":
    asyncio.run(main())