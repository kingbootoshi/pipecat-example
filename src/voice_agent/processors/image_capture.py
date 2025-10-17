from __future__ import annotations

import os
from datetime import datetime

import cv2
from PIL import Image

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..constants import SCREENSHOTS_DIR
from ..logging import logger


class ImageCaptureProcessor(FrameProcessor):
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self._webcam_marker = "__webcam__"
        self._webcam = None
        self._init_webcam()

    def _init_webcam(self):
        try:
            logger.info("📹 Initializing webcam...")
            self._webcam = cv2.VideoCapture(0)
            if not self._webcam.isOpened():
                logger.error("❌ Could not open webcam")
                self._webcam = None
            else:
                width = int(self._webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"✅ Webcam initialized: {width}x{height}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize webcam: {e}")
            self._webcam = None

    def _take_webcam_photo(self):
        if self._webcam is None or not self._webcam.isOpened():
            logger.warning("⚠️ Webcam not available, skipping webcam capture")
            return None

        try:
            ret, frame = self._webcam.read()
            if not ret:
                logger.error("❌ Failed to capture webcam frame")
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            raw_bytes = img.tobytes()
            return raw_bytes, img.size, img.mode
        except Exception as e:
            logger.error(f"❌ Error capturing webcam photo: {e}")
            return None

    def _remove_old_images(self):
        if hasattr(self._context, "_messages"):
            self._context._messages = [
                msg
                for msg in self._context._messages
                if not (
                    isinstance(msg.get("content"), list)
                    and any(
                        item.get("text", "").startswith(self._webcam_marker)
                        for item in msg.get("content", [])
                        if isinstance(item, dict)
                    )
                )
            ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info("🎥 Capturing webcam for LLM context")
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                webcam_result = self._take_webcam_photo()
                if not webcam_result:
                    logger.warning("⚠️ Webcam capture failed, skipping image context")
                    await self.push_frame(frame, direction)
                    return

                webcam_bytes, webcam_size, webcam_mode = webcam_result

                os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                try:
                    webcam_path = os.path.join(
                        SCREENSHOTS_DIR, f"webcam_{timestamp}.jpg"
                    )
                    debug_webcam = Image.frombytes(
                        webcam_mode, webcam_size, webcam_bytes
                    )
                    debug_webcam.save(webcam_path)
                    logger.debug(f"💾 Webcam saved to: {webcam_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")

                self._remove_old_images()

                self._context.add_image_frame_message(
                    image=webcam_bytes,
                    text=f"{self._webcam_marker} User webcam view",
                    size=webcam_size,
                    format=webcam_mode,
                )

                logger.info(f"✅ Webcam image added to context: {webcam_size}")
            except Exception as e:
                logger.error(f"❌ Failed to capture webcam: {e}")

        await self.push_frame(frame, direction)


__all__ = ["ImageCaptureProcessor"]

