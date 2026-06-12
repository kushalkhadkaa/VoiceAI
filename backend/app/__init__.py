"""Local-first Nepali/English voice assistant backend."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(REPO_ROOT / ".local" / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

__all__ = ["__version__"]

__version__ = "0.1.0"
