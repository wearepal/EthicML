"""Algorithms by Zafar et al. for Demographic Parity"""
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Dict, Any, Union
from abc import abstractmethod

import pandas as pd

from ethicml.algorithms.inprocess.installed_model import InstalledModel
from ethicml.utility.data_structures import DataTuple, TestTuple
from ethicml.preprocessing.adjust_labels import LabelBinarizer


class _ZafarAlgorithmBase(InstalledModel):
    def __init__(self):
        super().__init__(
            name="zafar",
            url="https://github.com/predictive-analytics-lab/fair-classification.git",
            module="fair-classification",
            file_name="train_pr.py",
        )

    @staticmethod
    def create_file_in_zafar_format(
        data: Union[DataTuple, TestTuple], file_path: Path, label_converter: LabelBinarizer
    ):
        """Save a DataTuple as a JSON file, which is extremely inefficient but what Zafar wants"""
        out: Dict[str, Any] = {}
        out["x"] = data.x.to_numpy().tolist()
        sens_attr = data.s.columns[0]
        out["sensitive"] = {}
        out["sensitive"][sens_attr] = data.s[sens_attr].to_numpy().tolist()
        if isinstance(data, DataTuple):
            data_converted = label_converter.adjust(data)
            class_attr = data.y.columns[0]
            out["class"] = (2 * data_converted.y[class_attr].to_numpy() - 1).tolist()
        else:
            zeros = pd.DataFrame([0 for _ in range(data.x.shape[0])], columns=['y'])
            out["class"] = zeros.to_numpy().tolist()
        with file_path.open("w") as out_file:
            json.dump(out, out_file)

    async def run_async(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        label_converter = LabelBinarizer()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.json"
            test_path = tmp_path / "test.json"
            self.create_file_in_zafar_format(train, train_path, label_converter)
            self.create_file_in_zafar_format(test, test_path, label_converter)
            predictions_path = tmp_path / "predictions.json"

            cmd = self._create_command_line(str(train_path), str(test_path), str(predictions_path))
            working_dir = self._module_path.resolve() / "disparate_impact" / "run-classifier"
            await self._call_script(cmd, cwd=working_dir)
            predictions = predictions_path.open().read()
            predictions = json.loads(predictions)

        predictions_correct = pd.DataFrame([0 if x == -1 else 1 for x in predictions])
        return label_converter.post_only_labels(predictions_correct)

    @abstractmethod
    def _create_command_line(self, train_name: str, test_name: str, predictions_name: str):
        pass


class ZafarBaseline(_ZafarAlgorithmBase):
    """Zafar without fairness"""

    def _create_command_line(self, train_name: str, test_name: str, predictions_name: str):
        return ["main.py", train_name, test_name, predictions_name, 'baseline', '0']

    @property
    def name(self):
        return "ZafarBaseline"


class ZafarAccuracy(_ZafarAlgorithmBase):
    """Zafar with fairness"""

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def _create_command_line(self, train_name: str, test_name: str, predictions_name: str):
        return ["main.py", train_name, test_name, predictions_name, 'gamma', str(self.gamma)]

    @property
    def name(self):
        return f"ZafarAccuracy, gamma={self.gamma}"


class ZafarFairness(_ZafarAlgorithmBase):
    """Zafar with fairness"""

    def __init__(self, c: float = 0.001):
        super().__init__()
        self._c = c

    def _create_command_line(self, train_name: str, test_name: str, predictions_name: str):
        return ["main.py", train_name, test_name, predictions_name, 'c', str(self._c)]

    @property
    def name(self):
        return f"ZafarFairness, c={self._c}"
