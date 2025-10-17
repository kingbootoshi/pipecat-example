from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from .constants import DEFAULT_CONFIG_FILE
from .logging import logger


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG_FILE
    if not os.path.exists(cfg_path):
        logger.error(f"Configuration file not found: {cfg_path}")
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    try:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {cfg_path}")
            return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse configuration file: {e}")
        raise


__all__ = ["load_config"]

