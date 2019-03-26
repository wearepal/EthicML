"""
Wrapper for calling Kamishima model
"""
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
        self.min_class_label = None  # minimum of the class labels

    @staticmethod
    def create_file_in_kamishima_format(data, file_path):
        """Create a text file with the data"""
        result = pd.concat([data.x, data.s, data.y], axis=1).to_numpy().astype(np.float64)
        np.savetxt(file_path, result)

    def write_data(self, train, test, tmp_path):
        train_name = str(tmp_path / "train.txt")
        test_name = str(tmp_path / "test.txt")
        self.create_file_in_kamishima_format(train, train_name)
        self.create_file_in_kamishima_format(test, test_name)
        self.min_class_label = train.y[train.y.columns[0]].min()
        return train_name, test_name

    def run_thread(self, train_paths, test_paths, tmp_path):
        model_name = str(tmp_path / "model")
        output_name = str(tmp_path / "output.txt")

        self._call_script([str(ROOT_DIR / self.repo_name / self.module / 'train_pr.py'),
                           '-e', str(self.eta),
                           '-i', train_paths,
                           '-o', model_name,
                           '--quiet'])
        self._call_script([str(ROOT_DIR / self.repo_name / self.module / 'predict_lr.py'),
                           '-i', test_paths,
                           '-m', model_name,
                           '-o', output_name,
                           '--quiet'])
        return output_name

    def load_output(self, output_path):
        output = np.loadtxt(output_path)

        predictions = output[:, 1].astype(np.float32)
        to_return = pd.DataFrame(predictions, columns=["preds"])

        if self.min_class_label is not None:
            return to_return.replace(to_return['preds'].min(), self.min_class_label)
        return to_return

    @property
    def name(self):
        return "Kamishima"
