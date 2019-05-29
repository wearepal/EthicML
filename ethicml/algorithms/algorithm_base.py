"""
Base class for Algorithms
"""
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from subprocess import check_call, CalledProcessError

import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a feather file"""
    with path.open('rb') as file:
        df = pd.read_feather(file)
    return df


class Algorithm(ABC):
    """Base class for Algorithms"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""


class AlgorithmAsync(ABC):
    """Base class of async methods. This class is meant to be used in conjuction with `Algorithm`"""

    @property
    def _executable(self) -> str:
        """
        Path to a (Python) executable

        By default, the Python executable that called this script is used.
        """
        return sys.executable

    def _call_script(self, cmd_args: List[str], env: Optional[Dict[str, str]] = None):
        """This function calls a (Python) script as a separate process

        An exception is thrown if the called script failed.

        Args:
            cmd_args: list of strings that are passed as commandline arguments to the executable
            env: environment variables specified as a dictionary; e.g. {"PATH": "/usr/bin"}
        """
        cmd = [self._executable] + cmd_args
        try:
            check_call(cmd, env=env)
        except CalledProcessError:
            raise RuntimeError(
                f'The script failed. Supplied arguments: {cmd_args} with exec: {self._executable}'
            )
