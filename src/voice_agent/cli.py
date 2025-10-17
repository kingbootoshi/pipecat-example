from __future__ import annotations

import argparse
import asyncio

from .app import VoiceAgent
from .serve import serve as serve_both


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
    p.add_argument(
        "--no-serve",
        action="store_true",
        help="Run pipeline only (skip web controller)",
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="Controller host")
    p.add_argument("--port", type=int, default=8710, help="Controller port")
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

    if args.no_serve:
        asyncio.run(agent.start())
    else:
        asyncio.run(serve_both(agent, host=args.host, port=args.port))


__all__ = ["main"]
