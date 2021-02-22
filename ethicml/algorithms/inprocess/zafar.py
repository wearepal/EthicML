"""Algorithms by Zafar et al. for Demographic Parity."""
import json
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, List, Union

import pandas as pd
from typing_extensions import Final

from ethicml.preprocessing.adjust_labels import LabelBinarizer
from ethicml.utility import DataTuple, Prediction, TestTuple

from .installed_model import InstalledModel

__all__ = ["ZafarAccuracy", "ZafarBaseline", "ZafarEqOdds", "ZafarEqOpp", "ZafarFairness"]


SUB_DIR_IMPACT = Path(".") / "disparate_impact" / "run-classifier"
SUB_DIR_MISTREAT = Path(".") / "disparate_mistreatment" / "run_classifier"

MAIN: Final = "main.py"


class _ZafarAlgorithmBase(InstalledModel):
    def __init__(self, name: str, sub_dir: Path):
        super().__init__(
            name=name,
            dir_name="zafar",
            url="https://github.com/predictive-analytics-lab/fair-classification.git",
            top_dir="fair-classification",
        )
        self._sub_dir = sub_dir

    @staticmethod
    def _create_file_in_zafar_format(
        data: Union[DataTuple, TestTuple], file_path: Path, label_converter: LabelBinarizer
    ) -> None:
        """Save a DataTuple as a JSON file, which is extremely inefficient but what Zafar wants."""
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
            out["class"] = [-1 for _ in range(data.x.shape[0])]
        with file_path.open("w") as out_file:
            json.dump(out, out_file)

    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        label_converter = LabelBinarizer()
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_path = tmp_path / "train.json"
            test_path = tmp_path / "test.json"
            self._create_file_in_zafar_format(train, train_path, label_converter)
            self._create_file_in_zafar_format(test, test_path, label_converter)
            predictions_path = tmp_path / "predictions.json"

            cmd = self._create_command_line(str(train_path), str(test_path), str(predictions_path))
            working_dir = self._code_path.resolve() / self._sub_dir
            await self._call_script(cmd, cwd=working_dir)
            predictions = predictions_path.open().read()
            predictions = json.loads(predictions)

        predictions_correct = pd.Series([0 if x == -1 else 1 for x in predictions])
        return Prediction(hard=label_converter.post_only_labels(predictions_correct))

    @abstractmethod
    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        pass


class ZafarBaseline(_ZafarAlgorithmBase):
    """Zafar without fairness."""

    def __init__(self) -> None:
        super().__init__(name="ZafarBaseline", sub_dir=SUB_DIR_IMPACT)

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "baseline", "0"]


class ZafarAccuracy(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, gamma: float = 0.5):
        super().__init__(name=f"ZafarAccuracy, γ={gamma}", sub_dir=SUB_DIR_IMPACT)
        self.gamma = gamma

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "gamma", str(self.gamma)]


class ZafarFairness(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, c: float = 0.001):
        super().__init__(name=f"ZafarFairness, c={c}", sub_dir=SUB_DIR_IMPACT)
        self._c = c

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [MAIN, train_name, test_name, predictions_name, "c", str(self._c)]


class ZafarEqOpp(_ZafarAlgorithmBase):
    """Zafar for Equality of Opportunity."""

    _mode: ClassVar[str] = "fnr"  # class level constant
    _base_name: ClassVar[str] = "ZafarEqOpp"

    def __init__(self, tau: float = 5.0, mu: float = 1.2, eps: float = 0.0001):
        super().__init__(name=f"{self._base_name}, τ={tau}, μ={mu}", sub_dir=SUB_DIR_MISTREAT)
        self._tau = tau
        self._mu = mu
        self._eps = eps

    def _create_command_line(
        self, train_name: str, test_name: str, predictions_name: str
    ) -> List[str]:
        return [
            MAIN,
            train_name,
            test_name,
            predictions_name,
            self._mode,
            str(self._tau),
            str(self._mu),
            str(self._eps),
        ]


class ZafarEqOdds(ZafarEqOpp):
    """Zafar for Equalised Odds."""

    _mode: ClassVar[str] = "fprfnr"
    _base_name: ClassVar[str] = "ZafarEqOdds"
