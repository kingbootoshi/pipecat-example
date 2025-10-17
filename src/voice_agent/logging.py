import sys
from loguru import logger


def setup_logging(level: str = "DEBUG") -> None:
    try:
        logger.remove(0)
    except Exception:
        # If default sink isn't present, ignore
        pass
    logger.add(sys.stderr, level=level.upper())


__all__ = ["setup_logging", "logger"]

