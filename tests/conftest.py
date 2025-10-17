import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# Ensure `src/` is on sys.path for tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load environment variables from .env if present
load_dotenv()

