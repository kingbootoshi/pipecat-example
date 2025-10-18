from __future__ import annotations

import os
from datetime import datetime
import base64
import io

import cv2
from PIL import Image

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..constants import SCREENSHOTS_DIR, WEBCAM_MARKER
from ..logging import logger


class ImageCaptureProcessor(FrameProcessor):
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self._webcam_marker = WEBCAM_MARKER
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
                # Request 480p (16:9) to balance quality and size.
                # Some cameras may ignore these; we will still rescale after capture.
                self._webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
                self._webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

            # Resize to 480p height (maintain aspect ratio).
            TARGET_HEIGHT = 480
            w, h = img.size
            scale = min(TARGET_HEIGHT / h, 1.0)
            if scale < 1.0:
                new_size = (max(1, int(w * scale)), TARGET_HEIGHT)
                img = img.resize(new_size, Image.BILINEAR)

            # Encode as highly-compressed JPEG bytes.
            buf = io.BytesIO()
            img.save(
                buf,
                format="JPEG",
                quality=40,            # better quality but still small
                optimize=True,
                progressive=True,
                subsampling=2,         # 4:2:0 chroma subsampling
            )
            jpeg_bytes = buf.getvalue()

            # Also return resized size for logging/debugging
            return jpeg_bytes, img.size, "JPEG"
        except Exception as e:
            logger.error(f"❌ Error capturing webcam photo: {e}")
            return None

    def _remove_old_images(self):
        # Remove only prior webcam-only messages so we don't accidentally
        # delete real user turns that may contain text.
        def _is_webcam_only_message(msg: dict) -> bool:
            if msg.get("role") != "user":
                return False
            content = msg.get("content")
            if not isinstance(content, list):
                return False
            has_marker_text = False
            for item in content:
                if not isinstance(item, dict):
                    return False
                t = item.get("type")
                if t == "text":
                    text_val = str(item.get("text", ""))
                    if text_val.startswith(self._webcam_marker):
                        has_marker_text = True
                    else:
                        # Found non-marker user text -> not webcam-only
                        return False
                elif t == "image_url":
                    # Allowed alongside marker text
                    continue
                else:
                    # Unknown content type -> keep message
                    return False
            return has_marker_text

        messages = self._context.get_messages()
        filtered_messages = []
        changed = False

        for message in messages:
            if isinstance(message, dict) and _is_webcam_only_message(message):
                changed = True
                continue
            filtered_messages.append(message)

        if changed:
            self._context.set_messages(filtered_messages)

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

                webcam_bytes, webcam_size, webcam_format = webcam_result

                os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                try:
                    webcam_path = os.path.join(SCREENSHOTS_DIR, f"webcam_{timestamp}.jpg")
                    with open(webcam_path, "wb") as f:
                        f.write(webcam_bytes)
                    logger.debug(
                        f"💾 Webcam saved to: {webcam_path} ({webcam_size[0]}x{webcam_size[1]}, {len(webcam_bytes)} bytes)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save debug image: {e}")

                self._remove_old_images()

                # Build a data URL so we can control JPEG quality/size.
                data_url = "data:image/jpeg;base64," + base64.b64encode(webcam_bytes).decode("utf-8")

                self._context.add_message(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{self._webcam_marker} User webcam view"},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                )

                logger.info(
                    f"✅ Webcam image added to context: {webcam_size[0]}x{webcam_size[1]} ~{len(webcam_bytes)} bytes"
                )
            except Exception as e:
                logger.error(f"❌ Failed to capture webcam: {e}")

        await self.push_frame(frame, direction)


__all__ = ["ImageCaptureProcessor"]
