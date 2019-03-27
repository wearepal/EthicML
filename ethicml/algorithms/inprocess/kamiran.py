"""Kamiran and Calders 2012"""
from pathlib import Path
from typing import List

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.interface import conventional_interface
from ethicml.algorithms.utils import PathTuple
from ethicml.implementations import kamiran


class Kamiran(InAlgorithm):
    """
    Kamiran and Calders 2012
    """
    def _run(self, train, test):
        return kamiran.train_and_predict(train, test)

    def _script_command(self, train_paths: PathTuple, test_paths: PathTuple,
                        pred_path: Path) -> (List[str]):
        script = ['-m', kamiran.train_and_predict.__module__]
        args = conventional_interface(train_paths, test_paths, pred_path)
        return script + args

    @property
    def name(self) -> str:
        return "Kamiran & Calders"
