"""Beutel's algorithm"""
from typing import Optional, List, Dict

from ethicml.common import ROOT_PATH
from .threaded_pre_algorithm import BasicTPA


class ThreadedBeutel(BasicTPA):
    """Beutel's adversarially learned fair representations"""
    def __init__(self,
                 fairness: str = "DI",
                 enc_size: Optional[List[int]] = None,
                 adv_size: Optional[List[int]] = None,
                 pred_size: Optional[List[int]] = None,
                 enc_activation="Sigmoid",
                 adv_activation="Sigmoid",
                 batch_size: int = 64,
                 y_loss="BCELoss",
                 s_loss="BCELoss",
                 epochs=50):
        # pylint: disable=too-many-arguments
        super().__init__("Beutel", str(ROOT_PATH.parent / "examples" / "beutel.py"))

        # convert all parameter values to lists of strings
        self.flags: Dict[str, List[str]] = {
            'fairness': [fairness],
            'enc_size': ["40"] if enc_size is None else [str(i) for i in enc_size],
            'adv_size': ["40"] if adv_size is None else [str(i) for i in adv_size],
            'pred_size': ["40"] if pred_size is None else [str(i) for i in pred_size],
            'enc_activation': [enc_activation],
            'adv_activation': [adv_activation],
            'batch_size': [str(batch_size)],
            'y_loss': [y_loss],
            's_loss': [s_loss],
            'epochs': [str(epochs)],
        }

    def _script_interface(self, train_paths, test_paths, for_train_path, for_test_path):
        """Generate the commandline arguments that are expected by the Beutel script"""
        flags_list: List[str] = []

        # paths to training and test data
        flags_list += self._path_tuple_to_cmd_args([train_paths, test_paths],
                                                   ['--train_', '--test_'])

        # paths to output files
        flags_list += ['--train_in', str(for_train_path), '--test_in', str(for_test_path)]

        # model parameters
        for key, values in self.flags.items():
            flags_list.append(f"--{key}")
            flags_list += values
        return flags_list
