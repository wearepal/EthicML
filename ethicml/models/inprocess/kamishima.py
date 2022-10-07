"""Wrapper for calling Kamishima model."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.models.inprocess.in_algorithm import InAlgorithm
from ethicml.models.inprocess.installed_model import InstalledModel
from ethicml.utility import DataTuple, HyperParamType, Prediction, TestTuple

__all__ = ["Kamishima"]


class _FitInfo(NamedTuple):
    """Information that is stored after the model has been fit."""

    min_class_label: int
    model_path: Path


class Kamishima(InstalledModel):
    """Model that calls Kamishima's code.

    Based on Algo-Fairness
    https://github.com/algofairness/fairness-comparison/blob/master/fairness/algorithms/kamishima/KamishimaAlgorithm.py

    :param eta: Tolerance.
    """

    def __init__(self, *, eta: float = 1.0):
        super().__init__(
            name="Kamishima",
            dir_name="kamishima",
            url="https://github.com/predictive-analytics-lab/kamfadm.git",
            top_dir="kamfadm",
        )
        self.eta = eta
        self._fit_info: _FitInfo | None = None

    @property
    @implements(InAlgorithm)
    def hyperparameters(self) -> HyperParamType:
        return {"eta": self.eta}

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_info = self._fit(train, tmp_path, seed)
            return self._predict(test, fit_info, tmp_path)

    @implements(InAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> Kamishima:
        with TemporaryDirectory() as tmpdir:
            self._fit_info = self._fit(train, Path(tmpdir), seed, model_dir=self._code_path)
        return self

    def _fit(
        self, train: DataTuple, tmp_path: Path, seed: int, model_dir: Path | None = None
    ) -> _FitInfo:
        train_path = tmp_path / "train.txt"
        _create_file_in_kamishima_format(train, train_path)
        min_class_label: int = train.y.min()
        model_path = (model_dir if model_dir is not None else tmp_path) / "model"

        script = self._code_path / "train_pr.py"
        cmd = [script, "-e", self.eta, "-i", train_path, "-o", model_path, "--quiet"]
        self.call_script([str(e) for e in cmd])
        return _FitInfo(min_class_label=min_class_label, model_path=model_path)

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        assert self._fit_info is not None, "call fit() before calling predict()"
        with TemporaryDirectory() as tmpdir:
            return self._predict(test, self._fit_info, Path(tmpdir))

    def _predict(self, test: TestTuple, fit_info: _FitInfo, tmp_path: Path) -> Prediction:
        test_path = tmp_path / "test.txt"
        _create_file_in_kamishima_format(test, test_path)
        output_path = str(tmp_path / "output.txt")
        script = self._code_path / "predict_lr.py"
        cmd = [script, "-i", test_path, "-m", fit_info.model_path, "-o", output_path, "--quiet"]
        self.call_script([str(e) for e in cmd])
        output = np.loadtxt(output_path)
        predictions = output[:, 1].astype(np.float32)
        # except RuntimeError:
        #     predictions = np.ones_like(test.y.to_numpy())

        to_return = pd.Series(predictions)
        to_return = to_return.astype(int)

        if to_return.min() != to_return.max():
            to_return = to_return.replace(to_return.min(), fit_info.min_class_label)
        return Prediction(hard=to_return)


def _create_file_in_kamishima_format(data: DataTuple | TestTuple, file_path: Path) -> None:
    """Create a text file with the data.

    :param data: Data to write to the file.
    :param file_path: Path to the file.
    """
    if isinstance(data, DataTuple):
        result = pd.concat([data.x, data.s, data.y], axis="columns").to_numpy().astype(np.float64)
    else:
        zeros = pd.DataFrame([0 for _ in range(data.x.shape[0])], columns=["y"])
        result = pd.concat([data.x, data.s, zeros], axis="columns").to_numpy().astype(np.float64)
    np.savetxt(file_path, result)
