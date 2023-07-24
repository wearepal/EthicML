"""Installable model.

This is a kind of complicated model, but it's incredibly useful.
Say you find a paper from a few years ago with code. It's not unreasonable that there might
be dependency clashes, python clashes, clashes galore. This approach downloads a model, runs it
in its own venv and makes everyone happy.
"""
from __future__ import annotations
from abc import ABC
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing_extensions import override

import git.cmd

from ethicml.models.inprocess.in_algorithm import InAlgorithm

from ..algorithm_base import SubprocessAlgorithmMixin

__all__ = ["InstalledModel"]


class InstalledModel(SubprocessAlgorithmMixin, InAlgorithm, ABC):
    """The model that does the magic.

    Download code from given URL and create Pip environment with Pipfile found in the code.

    :param name: Name of the model.
    :param dir_name: Where to download the code to (can be chosen freely).
    :param top_dir: Top directory of the repository where the Pipfile can be found (this is usually
        simply the last part of the repository URL).
    :param url: URL of the repository. (Default: None)
    :param executable: Path to a Python executable. (Default: None.
    :param use_pdm: If ``True``, will try to use pdm instead of pipenv. (Default: False)
    """

    def __init__(
        self,
        name: str,
        dir_name: str,
        top_dir: str,
        url: str | None = None,
        executable: list[str] | None = None,
        use_pdm: bool = False,
    ):
        # QUESTION: do we really need `store_dir`? we could also just clone the code into "."
        self._store_dir: Path = Path() / dir_name  # directory where code and venv are stored
        self._top_dir: str = top_dir

        if url is not None:
            # download code
            self._clone_directory(url)

        if executable is None:
            # create virtual environment
            self.__executable = self._create_venv(use_pdm=use_pdm)
        else:
            self.__executable = executable
        self.__name = name

    @property
    @override
    def name(self) -> str:
        return self.__name

    @property
    def _code_path(self) -> Path:
        """Path to where the code of the model is located."""
        return self._store_dir / self._top_dir

    @property
    def executable(self) -> list[str]:
        """Python executable from the virtualenv associated with the model."""
        return self.__executable

    def _clone_directory(self, url: str) -> None:
        """Clones the repo from `url` into `self._store_dir`.

        :param url: URL of the repository.
        """
        if not self._store_dir.exists():
            self._store_dir.mkdir()
            git.cmd.Git(self._store_dir).clone(url)

    def _create_venv(self, use_pdm: bool) -> list[str]:
        """Create a venv based on the Pipfile in the repository.

        :param use_pdm: Whether to use pdm instead of pipenv.
        :returns: Shell command to execute python scripts.
        :raises RuntimeError: If the required package manager isn't found.
        """
        if use_pdm:  # use pdm instead of pipenv
            pypackages_dir = self._code_path / "__pypackages__"
            if not pypackages_dir.exists():
                if shutil.which("pdm") is None:
                    raise RuntimeError("pdm must be installed")
                environ = os.environ.copy()
                environ["PDM_USE_VENV"] = "0"
                subprocess.run(
                    ["pdm", "install", "--no-self"], env=environ, check=True, cwd=self._code_path
                )
            return ["pdm", "run", "python"]

        venv_directory = self._code_path.resolve() / ".venv"
        if not venv_directory.exists():
            environ = os.environ.copy()
            environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
            environ["PIPENV_VENV_IN_PROJECT"] = "true"
            environ["PIPENV_YES"] = "true"
            environ["PIPENV_PIPFILE"] = str(self._code_path / "Pipfile")

            subprocess.run([sys.executable, "-m", "pipenv", "install"], env=environ, check=True)
        return [str(venv_directory / "bin" / "python")]

    def remove(self) -> None:
        """Remove the directory that we created in :meth:`_clone_directory()`."""
        try:
            shutil.rmtree(self._store_dir)
        except OSError as excep:
            print(f"Error: {excep.filename} - {excep.strerror}.")
