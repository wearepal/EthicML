import os
import subprocess
from pathlib import Path

import git

import shutil


from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


class VenvSVM(InAlgorithm):
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

    def create_venv(self, name):
        os.chdir(Path(f"./{name}/test_svm_module/"))
        os.environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
        os.environ["PIPENV_VENV_IN_PROJECT"] = "true"
        os.environ["PIPENV_YES"] = "true"

        venv_directory = Path(f"./.venv")
        if not os.path.exists(venv_directory):
            subprocess.check_call(["/Users/ot44/anaconda3/envs/test_env/bin/pipenv", "install"])

    def __init__(self, name: str, url: str):
        self.clone_directory(name, url)
        self.create_venv(name)
        super().__init__(executable="/Users/ot44/Development/EthicML/tests/oliver_git_svm/test_svm_module/.venv/bin/python")

    def run(self, train, test, sub_process: bool = False):
        return self.run_threaded(train, test)

    def run_thread(self, train_paths, test_paths, tmp_path):
        pred_path = tmp_path / "predictions.parquet"
        args = self._script_interface(train_paths, test_paths, pred_path)
        self._call_script("/Users/ot44/Development/EthicML/tests/oliver_git_svm/test_svm_module/SVMTWO.py", args)
        return pred_path

    def remove(self, name):
        os.chdir(Path(f"../.."))
        directory = Path(f"./{name}")
        try:
            shutil.rmtree(directory)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


