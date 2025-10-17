import os
from pipecat.services.assemblyai.stt import AssemblyAISTTService


def make_stt() -> AssemblyAISTTService:
    return AssemblyAISTTService(api_key=os.getenv("ASSEMBLYAI_API_KEY"))


__all__ = ["make_stt"]

