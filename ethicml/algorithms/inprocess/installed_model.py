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
from typing import List, Optional

import git

from .in_algorithm import InAlgorithmAsync

__all__ = ["InstalledModel"]


class InstalledModel(InAlgorithmAsync):
    """the model that does the magic."""

    def __init__(
        self,
        name: str,
        dir_name: str,
        top_dir: str,
        url: Optional[str] = None,
        executable: Optional[str] = None,
    ):
        """Download code from given URL and create Pip environment with Pipfile found in the code.

        Args:
            name: name of the model
            dir_name: where to download the code to (can be chosen freely)
            top_dir: top directory of the repository where the Pipfile can be found (this is usually
                     simply the last part of the repository URL)
            url: (optional) URL of the repository
            executable: (optional) path to a Python executable
        """
        # QUESTION: do we really need `store_dir`? we could also just clone the code into "."
        self._store_dir: Path = Path(".") / dir_name  # directory where code and venv are stored
        self._top_dir: str = top_dir

        if url is not None:
            # download code
            self._clone_directory(url)

        if executable is None:
            # create virtual environment
            self._create_venv()
            self.__executable = str(self._code_path.resolve() / ".venv" / "bin" / "python")
        else:
            self.__executable = executable
        super().__init__(name=name)

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

    def _create_venv(self) -> None:
        """Creates a venv based on the Pipfile in the repository."""
        venv_directory = self._code_path / ".venv"
        if not venv_directory.exists():
            environ = os.environ.copy()
            environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
            environ["PIPENV_VENV_IN_PROJECT"] = "true"
            environ["PIPENV_YES"] = "true"
            environ["PIPENV_PIPFILE"] = str(self._code_path / "Pipfile")

            subprocess.check_call([sys.executable, "-m", "pipenv", "install"], env=environ)

    def remove(self) -> None:
        """Removes the directory that we created in _clone_directory()."""
        try:
            shutil.rmtree(self._store_dir)
        except OSError as excep:
            print("Error: %s - %s." % (excep.filename, excep.strerror))

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        return []  # pylint was complaining when I didn't return anything here...
