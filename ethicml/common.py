"""Common variables / constants that make things run smoother."""
import os
from importlib import util
from pathlib import Path

__all__ = ["TORCH_AVAILABLE", "TORCHVISION_AVAILABLE", "ROOT_DIR", "ROOT_PATH"]

TORCH_AVAILABLE = util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = util.find_spec("torchvision") is not None

ROOT_DIR: str = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
ROOT_PATH: Path = Path(__file__).parent.resolve()
