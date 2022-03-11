"""Installable model.

This is a kind of complicated model, but it's incredibly useful.
Say you find a paper from a few years ago with code. It's not unreasonable that there might
be dependency clashes, python clashes, clashes galore. This approach downloads a model, runs it
in its own venv and makes everyone happy.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import git

from ..algorithm_base import SubprocessAlgorithmMixin
from .in_algorithm import InAlgorithm

__all__ = ["InstalledModel"]


class InstalledModel(SubprocessAlgorithmMixin, InAlgorithm):
    """The model that does the magic."""

    def __init__(
        self,
        name: str,
        dir_name: str,
        top_dir: str,
        is_fairness_algo: bool,
        url: Optional[str] = None,
        executable: Optional[str] = None,
        seed: int = 888,
        use_poetry: bool = False,
    ):
        """Download code from given URL and create Pip environment with Pipfile found in the code.

        Args:
            name: name of the model
            dir_name: where to download the code to (can be chosen freely)
            top_dir: top directory of the repository where the Pipfile can be found (this is usually
                     simply the last part of the repository URL)
            is_fairness_algo: if True, this object corresponds to an algorithm enforcing fairness
            url: (optional) URL of the repository
            executable: (optional) path to a Python executable
            seed: Random seed to use for reproducibility
            use_poetry: if True, will try to use poetry instead of pipenv
        """
        # QUESTION: do we really need `store_dir`? we could also just clone the code into "."
        self._store_dir: Path = Path(".") / dir_name  # directory where code and venv are stored
        self._top_dir: str = top_dir

        if url is not None:
            # download code
            self._clone_directory(url)

        if executable is None:
            # create virtual environment
            del use_poetry  # see https://github.com/python-poetry/poetry/issues/4055
            # self._create_venv(use_poetry=use_poetry)
            self._create_venv(use_poetry=False)
            self.__executable = str(self._code_path.resolve() / ".venv" / "bin" / "python")
        else:
            self.__executable = executable
        self.__name = name
        self.seed = seed
        self.is_fairness_algo = is_fairness_algo

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return self.__name

    @property
    def _code_path(self) -> Path:
        """Path to where the code of the model is located."""
        return self._store_dir / self._top_dir

    @property
    def _executable(self) -> str:
        return self.__executable

    def _clone_directory(self, url: str) -> None:
        """Clones the repo from `url` into `self._store_dir`."""
        if not self._store_dir.exists():
            self._store_dir.mkdir()
            git.Git(self._store_dir).clone(url)

    def _create_venv(self, use_poetry: bool) -> None:
        """Creates a venv based on the Pipfile in the repository."""
        venv_directory = self._code_path / ".venv"
        if not venv_directory.exists():
            if use_poetry and shutil.which("poetry") is not None:  # use poetry instead of pipenv
                environ = os.environ.copy()
                environ["POETRY_VIRTUALENVS_CREATE"] = "true"
                environ["POETRY_VIRTUALENVS_IN_PROJECT"] = "true"
                subprocess.run(
                    ["poetry", "install", "--no-root"], env=environ, check=True, cwd=self._code_path
                )
                return

            environ = os.environ.copy()
            environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
            environ["PIPENV_VENV_IN_PROJECT"] = "true"
            environ["PIPENV_YES"] = "true"
            environ["PIPENV_PIPFILE"] = str(self._code_path / "Pipfile")

            subprocess.run([sys.executable, "-m", "pipenv", "install"], env=environ, check=True)

    def remove(self) -> None:
        """Removes the directory that we created in _clone_directory()."""
        try:
            shutil.rmtree(self._store_dir)
        except OSError as excep:
            print(f"Error: {excep.filename} - {excep.strerror}.")
