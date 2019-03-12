import pdb
import pickle

import os
import subprocess
from pathlib import Path

import git

from os import listdir
from os.path import isfile, join
import importlib
import shutil
import sys


from ethicml.algorithms.algorithm_base import Algorithm


class VenvManager:
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
            subprocess.check_call(["/Users/ot44/anaconda3/bin/pipenv", "install"])

    def setup(self, name: str, url: str) -> Algorithm:
        self.clone_directory(name, url)
        self.create_venv(name)

        print("stop")
        # pdb.set_trace()
        subprocess.check_call(["/Users/ot44/Development/EthicML/tests/oliver_git_svm/test_svm_module/.venv/bin/python", "/Users/ot44/Development/EthicML/ethicml/ven_manager/module_getter.py"], cwd="/Users/ot44/Development/EthicML/tests/oliver_git_svm/test_svm_module")
        _class = pickle.load(open( "dumped_pickle.p", "rb" ))

        # module = importlib.import_module(f"{name}.test_svm_module.SVM")
        # _class = getattr(module, "SVMEXAMPLE")

        instance = _class()
        instance.executable = Path(__file__).parent.parent.parent / "tests" / name / "test_svm_module" /".venv/bin/python"
        return instance

    def remove(self, name):
        os.chdir(Path(f"../.."))
        directory = Path(f"./{name}")
        try:
            shutil.rmtree(directory)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


