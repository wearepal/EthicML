"""Common variables / constants that make things run smoother."""
from importlib import util
from pathlib import Path

__all__ = ["TORCH_AVAILABLE", "ROOT_PATH"]

TORCH_AVAILABLE = util.find_spec("torch") is not None

ROOT_PATH: Path = Path(__file__).parent.resolve()
