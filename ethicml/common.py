"""common variables / consts that make things run smoother"""

import os
from pathlib import Path

ROOT_DIR: str = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
ROOT_PATH: Path = Path(__file__).parent.resolve()
