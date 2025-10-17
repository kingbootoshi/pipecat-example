from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LLMResponseLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.response_buffer = ""
        self.is_streaming = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self.response_buffer += frame.text
            self.is_streaming = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.is_streaming and self.response_buffer.strip():
                print(f"🤖 AI Response: {self.response_buffer.strip()}")
            self.response_buffer = ""
            self.is_streaming = False

        await self.push_frame(frame, direction)


__all__ = ["LLMResponseLogger"]

