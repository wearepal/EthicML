import os
import subprocess
from pathlib import Path

import git

import shutil
from ethicml.common import ROOT_PATH

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


ROOT_DIR = str(ROOT_PATH.parent)


class VenvSVM(InAlgorithm):
    def __init__(self, name: str, url: str, module: str, file_name: str):
        self.repo_name = name
        self.module = module
        self.file_name = file_name
        self.url = url
        self.clone_directory(name, url)
        self.create_venv()
        super().__init__(executable=f"{ROOT_DIR}/{name}/{module}/.venv/bin/python")

    @property
    def name(self) -> str:
        return "venv SVM"

    def clone_directory(self, name, url):
        directory = Path(f"./{name}")
        if not os.path.exists(directory):
            os.makedirs(directory)
            git.Git(directory).clone(url)
        else:
            print("Repo found")

    def create_venv(self):
        environ = os.environ.copy()
        environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
        environ["PIPENV_VENV_IN_PROJECT"] = "true"
        environ["PIPENV_YES"] = "true"
        environ["LANG"] = 'en_GB.UTF-8'
        environ["PIPENV_PIPFILE"] = f'/{ROOT_DIR}/{self.repo_name}/{self.module}/Pipfile'

        venv_directory = Path(f"./{self.repo_name}/{self.module}/.venv")

        if not os.path.exists(venv_directory):
            subprocess.check_call("pipenv install",
                                  env=environ, shell=True)

    def _run(self, train, test):
        pass

    def run_thread(self, train_paths, test_paths, tmp_path):
        pred_path = tmp_path / "predictions.parquet"
        args = self._script_interface(train_paths, test_paths, pred_path)
        self._call_script(
            [f"/{ROOT_DIR}/{self.repo_name}/{self.module}/{self.file_name}"] + args)
        return pred_path

    def remove(self):
        directory = Path(f"./{self.repo_name}")
        try:
            shutil.rmtree(directory)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


