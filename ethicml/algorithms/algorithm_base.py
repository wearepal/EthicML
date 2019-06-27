"""
Base class for Algorithms
"""
import sys
from pathlib import Path
from abc import ABC, abstractmethod, ABCMeta
from typing import List, Optional, Dict, Coroutine, TypeVar, Any
import asyncio

import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a feather file"""
    with path.open("rb") as file:
        df = pd.read_feather(file)
    return df


class Algorithm(ABC):
    """Base class for Algorithms"""

    def __init__(self, *args, **kwargs):  # this seems really pointless but it makes mypy happy
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""


class AlgorithmAsync(Algorithm, metaclass=ABCMeta):
    """Base class of async methods. This class is meant to be used in conjuction with `Algorithm`"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""

    @property
    def _executable(self) -> str:
        """
        Path to a (Python) executable

        By default, the Python executable that called this script is used.
        """
        return sys.executable

    async def _call_script(self, cmd_args: List[str], env: Optional[Dict[str, str]] = None):
        """This function calls a (Python) script as a separate process

        An exception is thrown if the called script failed.

        Args:
            cmd_args: list of strings that are passed as commandline arguments to the executable
            env: environment variables specified as a dictionary; e.g. {"PATH": "/usr/bin"}
        """
        process = await asyncio.create_subprocess_exec(  # wait for process creation to finish
            self._executable,
            *cmd_args,
            stderr=asyncio.subprocess.PIPE,  # we capture the stderr for errors
            env=env,
        )
        # print(f"Started: {cmd_args!r} (pid = {process.pid})")

        try:  # wait for process itself to finish
            one_hour = 3600
            _, stderr = await asyncio.wait_for(process.communicate(), one_hour)
        except asyncio.TimeoutError as error:
            raise RuntimeError("The script timed out.") from error

        if process.returncode == 0:
            print(f"Success: {cmd_args!r} (pid = {process.pid})")
        else:
            print(f"Failure: {cmd_args!r} (pid = {process.pid})")
            if stderr:
                print(stderr.decode().strip())
            raise RuntimeError(
                f"The script failed. Supplied arguments: {cmd_args} with exec: {self._executable}"
            )


_T = TypeVar("_T")


def run_blocking(promise: Coroutine[Any, Any, _T]) -> _T:
    """Run an asynchronous process as a blocking process"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(promise)
