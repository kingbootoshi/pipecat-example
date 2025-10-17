from __future__ import annotations

import argparse
import asyncio

from .app import VoiceAgent


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice-agent", description="Run Voice Agent")
    p.add_argument("run", nargs="?", default="run", help=argparse.SUPPRESS)
    p.add_argument("--config", type=str, default=None, help="Path to config.yml")
    p.add_argument("--no-vision", action="store_true", help="Disable webcam context")
    p.add_argument("--no-memory", action="store_true", help="Disable conversation memory")
    p.add_argument(
        "--allow-interruptions",
        action="store_true",
        help="Allow user to interrupt TTS",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    agent = VoiceAgent(
        config_path=args.config,
        vision=not args.no_vision,
        memory=not args.no_memory,
        allow_interruptions=args.allow_interruptions,
        log_level=args.log_level,
    )

    asyncio.run(agent.start())


__all__ = ["main"]

