"""Kamiran and Calders 2012"""
from .pre_algorithm import PreAlgorithm
from .interface import flag_interface


class Kamiran(PreAlgorithm):
    """
    Kamiran and Calders 2012
    """
    def _run(self, train, test):
        from ...implementations import kamiran
        return kamiran.compute_weights(train, test)

    def _script_command(self, train_paths, test_paths, new_train_path, new_test_path):
        args = flag_interface(train_paths, test_paths, new_train_path, new_test_path, {})
        return ['-m', 'ethicml.implementations.kamiran'] + args

    @property
    def name(self) -> str:
        return "Kamiran & Calders"
