"""Base class for Algorithms."""
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["Algorithm", "SubprocessAlgorithmMixin"]


class Algorithm(ABC):
    """Base class for Algorithms."""

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self.get_name()

    # a method is nicer to implement than a property because you can use the @implements decorator
    @abstractmethod
    def get_name(self) -> str:
        """Name of the algorithm."""


class SubprocessAlgorithmMixin(ABC):  # pylint: disable=too-few-public-methods
    """Mixin for running algorithms in a subprocess, to be used with :class:`Algorithm`."""

    @property
    def executable(self) -> str:
        """Path to a (Python) executable.

        By default, the Python executable that called this script is used.
        """
        return sys.executable

    def call_script(
        self, cmd_args: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None
    ) -> None:
        """Call a (Python) script as a separate process.

        An exception is thrown if the called script failed.

        :param cmd_args: List of strings that are passed as commandline arguments to the executable.
        :param env: Environment variables specified as a dictionary; e.g. ``{"PATH": "/usr/bin"}``.
        :param cwd: If not None, change working directory to the given path before running command.
        """
        two_hours = 60 * 60 * 2  # 60secs * 60mins * 2 hours
        try:
            process = subprocess.run(  # wait for process creation to finish
                [self.executable] + cmd_args,
                capture_output=True,
                env=env,
                cwd=cwd,
                timeout=two_hours,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("The script timed out.") from error

        if process.returncode != 0:
            print(f"Failure: {cmd_args!r}")
            stderr = process.stderr
            if stderr:
                print(stderr.decode().strip())
            raise RuntimeError(
                f"The script failed. Supplied arguments: {cmd_args} with exec: {self.executable}"
            )
