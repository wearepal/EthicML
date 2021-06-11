"""Wrapper for calling Kamishima model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import pandas as pd
from kit import implements

from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import InAlgorithmAsync
from .installed_model import InstalledModel

__all__ = ["Kamishima"]


class Kamishima(InstalledModel):
    """Model that calls Kamishima's code.

    Based on Algo-Fairness
    https://github.com/algofairness/fairness-comparison/blob/master/fairness/algorithms/kamishima/KamishimaAlgorithm.py
    """

    def __init__(self, eta: float = 1.0):
        super().__init__(
            name="Kamishima",
            dir_name="kamishima",
            url="https://github.com/predictive-analytics-lab/kamfadm.git",
            top_dir="kamfadm",
        )
        self.eta = eta

    @staticmethod
    def create_file_in_kamishima_format(data: Union[DataTuple, TestTuple], file_path: str) -> None:
        """Create a text file with the data."""
        if isinstance(data, DataTuple):
            result = (
                pd.concat([data.x, data.s, data.y], axis="columns").to_numpy().astype(np.float64)
            )
        else:
            zeros = pd.DataFrame([0 for _ in range(data.x.shape[0])], columns=["y"])
            result = (
                pd.concat([data.x, data.s, zeros], axis="columns").to_numpy().astype(np.float64)
            )
        np.savetxt(file_path, result)

    @implements(InAlgorithmAsync)
    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = str(tmp_path / "train.txt")
            test_path = str(tmp_path / "test.txt")
            self.create_file_in_kamishima_format(train, train_path)
            self.create_file_in_kamishima_format(test, test_path)
            min_class_label = train.y[train.y.columns[0]].min()
            model_path = str(tmp_path / "model")
            output_path = str(tmp_path / "output.txt")

            # try:
            await self._call_script(
                [
                    str(self._code_path / "train_pr.py"),
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
                    str(self._code_path / "predict_lr.py"),
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

        to_return = pd.Series(predictions)
        to_return = to_return.astype(int)

        if to_return.min() != to_return.max():
            to_return = to_return.replace(to_return.min(), min_class_label)
        return Prediction(hard=to_return)
