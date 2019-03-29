"""
Wrapper for calling Kamishima model
"""
from pathlib import Path
from tempfile import TemporaryDirectory

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

    @staticmethod
    def create_file_in_kamishima_format(data, file_path):
        """Create a text file with the data"""
        result = pd.concat([data.x, data.s, data.y], axis=1).to_numpy().astype(np.float64)
        np.savetxt(file_path, result)

    def run(self, train, test, _=False):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = str(tmp_path / "train.txt")
            test_path = str(tmp_path / "test.txt")
            self.create_file_in_kamishima_format(train, train_path)
            self.create_file_in_kamishima_format(test, test_path)
            min_class_label = train.y[train.y.columns[0]].min()
            model_path = str(tmp_path / "model")
            output_path = str(tmp_path / "output.txt")

            self._call_script([str(ROOT_DIR / self.repo_name / self.module / 'train_pr.py'),
                               '-e', str(self.eta),
                               '-i', train_path,
                               '-o', model_path,
                               '--quiet'])
            self._call_script([str(ROOT_DIR / self.repo_name / self.module / 'predict_lr.py'),
                               '-i', test_path,
                               '-m', model_path,
                               '-o', output_path,
                               '--quiet'])

            output = np.loadtxt(output_path)

        predictions = output[:, 1].astype(np.float32)
        to_return = pd.DataFrame(predictions, columns=["preds"])

        return to_return.replace(to_return['preds'].min(), min_class_label)

    @property
    def name(self):
        return "Kamishima"
