"""
This is a kind of complicated model, but it's incredibly useful.
Say you find a papaer form a few years ago with code. It's not unreasonable that there might
be dependency clashes, python clashes, clashes galore. This approach downloads a model, runs it
in it's own venv and makes everyone happy.
"""
import os
from pathlib import Path
from typing import List, Optional
import shutil
import subprocess

import git

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.utility.data_structures import PathTuple, TestPathTuple


class InstalledModel(InAlgorithmAsync):
    """the model that does the magic"""

    def __init__(
        self,
        dir_name: str,
        top_dir: str,
        url: Optional[str] = None,
        executable: Optional[str] = None,
    ):
        """
        Download code from given URL and create a Pip environment with the Pipfile found in the code

        Args:
            dir_name: where to download the code to
            top_dir: top directory of the repository where the Pipfile can be found
            url: (optional) URL of the repository
            executable: (optional) path to Python executable
        """
        self._dir_name: str = dir_name
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
        super().__init__()

    @property
    def _code_path(self) -> Path:
        """Path to where the code of the model is located"""
        return Path(".") / self._dir_name / self._top_dir

    @property
    def _executable(self) -> str:
        return self.__executable

    def _clone_directory(self, url: str) -> None:
        """
        Clones the repo from `url` into `dir_name`
        """
        directory = Path(".") / self._dir_name
        if not directory.exists():
            directory.mkdir()
            git.Git(directory).clone(url)

    def _create_venv(self) -> None:
        """
        Creates a venv based on the repos Pipfile
        """
        environ = os.environ.copy()
        environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
        environ["PIPENV_VENV_IN_PROJECT"] = "true"
        environ["PIPENV_YES"] = "true"
        environ["PIPENV_PIPFILE"] = str(self._code_path / "Pipfile")

        venv_directory = self._code_path / ".venv"

        if not venv_directory.exists():
            subprocess.check_call("pipenv install", env=environ, shell=True)

    def remove(self) -> None:
        """
        Removes the directory that we created in _clone_directory()
        """
        directory = Path(".") / self._dir_name
        try:
            shutil.rmtree(directory)
        except OSError as excep:
            print("Error: %s - %s." % (excep.filename, excep.strerror))

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        return []  # pylint was complaining when I didn't return anything here...

    @property
    def name(self) -> str:
        raise NotImplementedError
