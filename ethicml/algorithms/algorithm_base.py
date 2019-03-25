"""
Base class for Algorithms
"""
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from subprocess import check_call, CalledProcessError

import pandas as pd

from .utils import PathTuple


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a feather file"""
    with path.open('rb') as file:
        df = pd.read_feather(file)
    return df


class Algorithm(ABC):
    """Base class for Algorithms"""
    def __init__(self, executable: Optional[str] = None,
                 hyperparams: Dict[str, float] = None):
        """Constructor

        Args:
            executable: (optional) path to a (Python) executable. If not provided, the Python
                        executable that called this script is used.
        """
        # self._name = name
        if executable is None:
            # use the python executable that this script was called with
            executable = sys.executable
        self.executable: str = executable
        self.hyperparams = hyperparams

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""
        raise NotImplementedError()

    def _call_script(self, cmd_args: List[str], env: Optional[Dict[str, str]] = None):
        """This function calls a (Python) script as a separate process

        An exception is thrown if the called script failed.

        Args:
            cmd_args: list of strings that are passed as commandline arguments to the executable
            env: environment variables specified as a dictionary; e.g. {"PATH": "/usr/bin"}
        """
        cmd = [self.executable] + cmd_args
        print("cmd =", str(cmd))
        print("exec =", str(self.executable))
        try:
            check_call(cmd, env=env)
        except CalledProcessError:
            raise RuntimeError(f'The script failed. Supplied arguments: {cmd_args}')

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

    @staticmethod
    def _load_output(file_path: Path) -> pd.DataFrame:
        """Load a dataframe from a feather file"""
        with file_path.open('rb') as file_obj:
            df = pd.read_feather(file_obj)
        return df
