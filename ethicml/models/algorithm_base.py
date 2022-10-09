"""Base class for Algorithms."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import sys

__all__ = ["Algorithm", "SubprocessAlgorithmMixin"]


class Algorithm(ABC):
    """Base class for Algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
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
        self, cmd_args: list[str], env: dict[str, str] | None = None, cwd: Path | None = None
    ) -> None:
        """Call a (Python) script as a separate process.

        :param cmd_args: List of strings that are passed as commandline arguments to the executable.
        :param env: Environment variables specified as a dictionary; e.g. ``{"PATH": "/usr/bin"}``.
        :param cwd: If not None, change working directory to the given path before running command.
        :raises RuntimeError: If the called script failed or timed out.
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
