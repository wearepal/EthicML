"""
Base class for Algorithms
"""
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
from subprocess import check_call, CalledProcessError

import pandas as pd

from .utils import PathTuple


class Algorithm(ABC):
    """Base class for Algorithms"""
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""
        raise NotImplementedError()


class ThreadedAlgorithm(ABC):
    """Base class for algorithms that run in their own thread"""
    def __init__(self, name: str, executable: Optional[str] = None):
        """Constructor

        Args:
            executable: (optional) path to a (Python) executable. If not provided, the Python
                        executable that called this script is used.
        """
        self._name = name
        if executable is None:
            # use the python executable that this script was called with
            executable = sys.executable
        self.executable: str = executable

    @property
    def name(self) -> str:
        """Name of the algorithm"""
        return self._name

    def _call_script(self, script: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """This function calls a (Python) script as a separate process

        An exception is thrown if the called script failed.

        Args:
            script: path to the (Python) script (if this is empty, then it is not used)
            args: list of strings that are passed as commandline arguments to the script
            env: environment variables specified as a dictionary; e.g. {"PATH": "/usr/bin"}
        """
        cmd = [self.executable]
        if script:
            cmd.append(script)
        cmd += args
        try:
            check_call(cmd, env=env)
        except CalledProcessError:
            raise RuntimeError(f'The script "{script}" failed. Supplied arguments: {args}')

    @staticmethod
    def _load_output(file_path: Path) -> pd.DataFrame:
        """Load a dataframe from a parquet file"""
        with file_path.open('rb') as file_obj:
            df = pd.read_parquet(file_obj)
        return df

    @staticmethod
    def _path_tuple_to_cmd_args(path_tuples: List[PathTuple], prefixes: List[str]) -> List[str]:
        """Convert the path tuples to a list of commandline arguments

        The list of prefixes must have the same length as the list of path tuples. Each path tuple
        is associated with one prefix. If the prefix for the path tuple "pt" is "--data_", then the
        following elements are added to the output list:
            ['--data_x', '<content of pt.x>', '--data_s', '<content of pt.s>', '--data_y',
             '<content of pt.y>']
        """
        args_list: List[str] = []
        for path_tuple, prefix in zip(path_tuples, prefixes):
            for key, path in path_tuple._asdict().items():
                args_list += [f"{prefix}{key}", str(path)]
        return args_list
