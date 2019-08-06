"""
Wrapper for calling Kamishima model
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ethicml.algorithms.inprocess.installed_model import InstalledModel
from ethicml.utility.data_structures import DataTuple, TestTuple


class Kamishima(InstalledModel):
    """
    Model that calls Kamishima's code. Based on Algo-Fairness
    https://github.com/algofairness/fairness-comparison/blob/master/fairness/algorithms/kamishima/KamishimaAlgorithm.py
    """

    def __init__(self, eta=1.0):
        super().__init__(
            name="kamishima",
            url="https://github.com/predictive-analytics-lab/kamfadm.git",
            module="kamfadm",
            file_name="train_pr.py",
        )
        self.eta = eta

    @staticmethod
    def create_file_in_kamishima_format(data: DataTuple, file_path: str):
        """Create a text file with the data"""

        result = pd.concat([data.x, data.s, data.y], axis="columns").to_numpy().astype(np.float64)
        np.savetxt(file_path, result)

    async def run_async(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = str(tmp_path / "train.txt")
            test_path = str(tmp_path / "test.txt")
            self.create_file_in_kamishima_format(train, train_path)
            # TODO: we should fill the y column with dummy values for the test data
            self.create_file_in_kamishima_format(test, test_path)
            min_class_label = train.y[train.y.columns[0]].min()
            model_path = str(tmp_path / "model")
            output_path = str(tmp_path / "output.txt")

            # try:
            await self._call_script(
                [
                    str(self._module_path / "train_pr.py"),
                    "-e",
                    str(self.eta),
                    "-i",
                    train_path,
                    "-o",
                    model_path,
                    "--quiet",
                ]
            )

            await self._call_script(
                [
                    str(self._module_path / "predict_lr.py"),
                    "-i",
                    test_path,
                    "-m",
                    model_path,
                    "-o",
                    output_path,
                    "--quiet",
                ]
            )
            output = np.loadtxt(output_path)
            predictions = output[:, 1].astype(np.float32)
            # except RuntimeError:
            #     predictions = np.ones_like(test.y.to_numpy())

        to_return = pd.DataFrame(predictions, columns=["preds"])

        if to_return["preds"].min() != to_return["preds"].max():
            to_return = to_return.replace(to_return["preds"].min(), min_class_label)
        return to_return

    @property
    def name(self):
        return "Kamishima"
