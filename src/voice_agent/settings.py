import os
from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv(override=True)


def missing_required_keys() -> list[str]:
    required = [
        "ASSEMBLYAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ELEVENLABS_API_KEY",
    ]
    return [k for k in required if not os.getenv(k)]


__all__ = ["load_env", "missing_required_keys"]

