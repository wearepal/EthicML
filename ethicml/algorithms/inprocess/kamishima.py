"""
Wrapper for calling Kamishima model
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from ethicml.algorithms.inprocess.installed_model import InstalledModel, ROOT_DIR


class Kamishima(InstalledModel):
    """
    Model that calls Kamishima's code. Based on Algo-Fairness
    https://github.com/algofairness/fairness-comparison/blob/master/fairness/algorithms/kamishima/KamishimaAlgorithm.py
    """
    def __init__(self, eta=1.0):
        super().__init__(name="kamishima",
                         url="https://github.com/predictive-analytics-lab/kamfadm.git",
                         module="kamfadm",
                         file_name="train_pr.py")
        self.eta = eta

    def _run(self, train, test):
        pass

    @staticmethod
    def create_file_in_kamishima_format(data, file_path):
        """

        Args:
            data:
            file_path:

        Returns:

        """
        y = data.y
        s = data.s
        x = data.x

        result = pd.concat([x, s, y], axis=1).to_numpy()
        result = result.astype(np.float64)

        np.savetxt(file_path, result)

    def run_threaded(self, train, test):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            class_type = type(train.y.values[0].item())

            model_name = str(tmp_path / "model")
            output_name = str(tmp_path / "output.txt")
            train_name = str(tmp_path / "train.txt")
            test_name = str(tmp_path / "test.txt")
            self.create_file_in_kamishima_format(train, train_name)
            self.create_file_in_kamishima_format(test, test_name)

            self._call_script([f'/{ROOT_DIR}/{self.repo_name}/{self.module}/train_pr.py',
                               '-e', str(self.eta),
                               '-i', train_name,
                               '-o', model_name,
                               '--quiet'])
            self._call_script([f'/{ROOT_DIR}/{self.repo_name}/{self.module}/predict_lr.py',
                               '-i', test_name,
                               '-m', model_name,
                               '-o', output_name,
                               '--quiet'])

            output = np.loadtxt(output_name)

        predictions = output[:, 1]
        predictions_correct = [class_type(x) for x in predictions]

        to_return = pd.DataFrame(predictions_correct, columns=["preds"])

        min_class_label = train.y[train.y.columns[0]].min()
        to_return = to_return.replace(to_return['preds'].min(), min_class_label)


        return to_return


    @property
    def name(self):
        return "Kamishima"
