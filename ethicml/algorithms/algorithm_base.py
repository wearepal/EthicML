"""Base class for Algorithms."""
import subprocess
import sys
from abc import ABC, ABCMeta
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["Algorithm", "AlgorithmAsync"]


class Algorithm(ABC):
    """Base class for Algorithms."""

    def __init__(self, name: str, seed: int):
        """Base constructor for the Algorithm class.

        Args:
            name: name of the algorithm
            seed: seed for the random number generator
        """
        self.__name = name
        self.__seed = seed

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self.__name

    @property
    def seed(self) -> int:
        """Seed for the random number generator."""
        return self.__seed


class AlgorithmAsync(metaclass=ABCMeta):  # pylint: disable=too-few-public-methods
    """Base class of async methods; meant to be used in conjuction with :class:`Algorithm`."""

    model_dir: Path

    @property
    def _executable(self) -> str:
        """Path to a (Python) executable.

        By default, the Python executable that called this script is used.
        """
        return sys.executable

    def _call_script(
        self, cmd_args: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None
    ) -> None:
        """This function calls a (Python) script as a separate process.

        An exception is thrown if the called script failed.

        Args:
            cmd_args: list of strings that are passed as commandline arguments to the executable
            env: environment variables specified as a dictionary; e.g. {"PATH": "/usr/bin"}
            cwd: if not None, change working directory to the given path before running command
        """
        one_hour = 3600
        try:
            process = subprocess.run(  # wait for process creation to finish
                [self._executable] + cmd_args,
                capture_output=True,
                env=env,
                cwd=cwd,
                timeout=one_hour,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("The script timed out.") from error

        if process.returncode != 0:
            print(f"Failure: {cmd_args!r}")
            stderr = process.stderr
            if stderr:
                print(stderr.decode().strip())
            raise RuntimeError(
                f"The script failed. Supplied arguments: {cmd_args} with exec: {self._executable}"
            )
