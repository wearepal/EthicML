import os
import subprocess
from pathlib import Path

import git

import shutil


from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


class VenvSVM(InAlgorithm):
    def __init__(self, name: str, url: str):
        self.clone_directory(name, url)
        self.create_venv(name)
        super().__init__(executable="/Users/ot44/Development/EthicML/oliver_git_svm/test_svm_module/.venv/bin/python")

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
        environ = os.environ.copy()
        environ["PIPENV_IGNORE_VIRTUALENVS"] = "1"
        environ["PIPENV_VENV_IN_PROJECT"] = "true"
        environ["PIPENV_YES"] = "true"
        environ["LANG"] = 'en_GB.UTF-8'
        environ["PIPENV_PIPFILE"] = '/Users/ot44/Development/EthicML/oliver_git_svm/test_svm_module/Pipfile'

        venv_directory = Path(f"./{name}/test_svm_module/.venv")

        if not os.path.exists(venv_directory):
            subprocess.check_call(["/Users/ot44/anaconda3/envs/test_env/bin/pipenv", "install"],
                                  env=environ)

    def _run(self, train, test):
        pass

    def run_thread(self, train_paths, test_paths, tmp_path):
        pred_path = tmp_path / "predictions.parquet"
        args = self._script_interface(train_paths, test_paths, pred_path)
        self._call_script(
            ["/Users/ot44/Development/EthicML/oliver_git_svm/test_svm_module/SVMTWO.py"] + args)
        return pred_path

    def remove(self, name):
        directory = Path(f"./{name}")
        try:
            shutil.rmtree(directory)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


