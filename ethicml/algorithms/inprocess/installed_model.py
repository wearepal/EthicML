"""
This is a kind of complicated model, but it's incredibly useful.
Say you find a papaer form a few years ago with code. It's not unreasonable that there might
be dependency clashes, python clashes, clashes galore. This approach downloads a model, runs it
in it's own venv and makes everyone happy.
"""
import os
from pathlib import Path
from typing import List, Union, overload, Optional
import shutil
import subprocess

import git

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.utility.data_structures import PathTuple, TestPathTuple


class InstalledModel(InAlgorithmAsync):
    """ the model that does the magic"""

    # pylint: disable=function-redefined,super-init-not-called,unused-argument

    @overload
    def __init__(self, name: str, url: str, module: str, file_name: str):
        ...

    @overload
    def __init__(self, name: str, url: str, module: str, file_name: Path, executable: str):
        ...

    def __init__(
        self,
        name: str,
        url: str,
        module: str,
        file_name: Union[str, Path],
        executable: Optional[str] = None,
    ):
        """
        The behavior of this class depends on whether a full file path is given in `file_name`. If
        yes, then it is assumed that the code is already there and it is not downloaded. In this
        case it is also assumed that the user already has a suitable python enviroment and no new
        environment is created.
        """
        self.script_path: Path
        self.__executable: str

        # check whether we have been given a complete path
        if isinstance(file_name, Path):
            self.script_path = file_name
            # if given a complete path, we also need the executable, because we don't have a Pipfile
            if executable is None:
                raise ValueError("When full file path is specified, an executable has to be given.")
            self.__executable = executable
        else:
            # no full path specified => we download the code ourselves
            self.url = url
            self.repo_name = name
            self.module = module

            # download code and set script path
            self.clone_directory()
            self.script_path = self._module_path / file_name

            # determine which executable to use
            if executable is not None:
                self.__executable = executable
            else:
                # no executable specified => we create our own environment
                self.create_venv()
                self.__executable = str(self._module_path / ".venv" / "bin" / "python")
        super().__init__()

    @property
    def _module_path(self) -> Path:
        return Path(".") / self.repo_name / self.module

    @property
    def _executable(self) -> str:
        return self.__executable

    @property
    def name(self) -> str:
        return self.module

    def clone_directory(self) -> None:
        """
        Clones the repo
        """
        directory = Path(".") / self.repo_name
        if not os.path.exists(directory):
            os.makedirs(directory)
            git.Git(directory).clone(self.url)

    def create_venv(self) -> None:
        """
        Creates a venv based on the repos Pipfile
        """
        environ = os.environ.copy()
        environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
        environ["PIPENV_VENV_IN_PROJECT"] = "true"
        environ["PIPENV_YES"] = "true"
        environ["PIPENV_PIPFILE"] = str(self._module_path / "Pipfile")

        venv_directory = self._module_path / ".venv"

        if not os.path.exists(venv_directory):
            subprocess.check_call("pipenv install", env=environ, shell=True)

    def _script_command(
        self, train_paths: PathTuple, test_paths: TestPathTuple, pred_path: Path
    ) -> (List[str]):
        args = conventional_interface(train_paths, test_paths, pred_path)
        return [str(self.script_path)] + args

    def remove(self) -> None:
        """
        Removes the directory that we created in clone_directory
        """
        directory = Path(".") / self.repo_name
        try:
            shutil.rmtree(directory)
        except OSError as excep:
            print("Error: %s - %s." % (excep.filename, excep.strerror))
