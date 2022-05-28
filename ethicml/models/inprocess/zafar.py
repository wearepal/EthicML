"""Algorithms by Zafar et al. for Demographic Parity."""
import json
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Union
from typing_extensions import Final

import pandas as pd
from ranzen import implements

from ethicml.preprocessing.adjust_labels import LabelBinarizer
from ethicml.utility import DataTuple, Prediction, TestTuple

from .in_algorithm import HyperParamType, InAlgorithm
from .installed_model import InstalledModel

__all__ = ["ZafarAccuracy", "ZafarBaseline", "ZafarEqOdds", "ZafarEqOpp", "ZafarFairness"]


SUB_DIR_IMPACT: Final = Path(".") / "disparate_impact" / "run-classifier"
SUB_DIR_MISTREAT: Final = Path(".") / "disparate_mistreatment" / "run_classifier"


class FitParams(NamedTuple):
    model_path: Path
    label_converter: LabelBinarizer


class _ZafarAlgorithmBase(InstalledModel):
    def __init__(self, name: str, sub_dir: Path):
        super().__init__(
            name=name,
            dir_name="zafar",
            url="https://github.com/predictive-analytics-lab/fair-classification.git",
            top_dir="fair-classification",
            use_poetry=True,
        )
        self._sub_dir = sub_dir
        self._fit_params: Optional[FitParams] = None

    @staticmethod
    def _create_file_in_zafar_format(
        data: Union[DataTuple, TestTuple], file_path: Path, label_converter: LabelBinarizer
    ) -> None:
        """Save a DataTuple as a JSON file, which is extremely inefficient but what Zafar wants.

        :param data: DataTuple to save.
        :param file_path: Path to save to.
        :param label_converter: Instance of a LabelBinarizer to convert labels to Zafar's format.
        """
        out: Dict[str, Any] = {'x': data.x.to_numpy().tolist(), "sensitive": {}}
        out["sensitive"][data.s.name] = data.s.to_numpy().tolist()
        if isinstance(data, DataTuple):
            data_converted = label_converter.adjust(data)
            out["class"] = (2 * data_converted.y.to_numpy() - 1).tolist()
        else:
            out["class"] = [-1 for _ in range(data.x.shape[0])]
        with file_path.open("w") as out_file:
            json.dump(out, out_file)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_params = self._fit(train, tmp_path, seed)
            return self._predict(test, tmp_path, fit_params)

    @implements(InAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> "_ZafarAlgorithmBase":
        with TemporaryDirectory() as tmpdir:
            self._fit_params = self._fit(
                train, tmp_path=Path(tmpdir), seed=seed, model_dir=self._code_path
            )
        return self

    @implements(InAlgorithm)
    def predict(self, test: TestTuple) -> Prediction:
        assert self._fit_params is not None, "call fit() first"
        with TemporaryDirectory() as tmpdir:
            return self._predict(test, tmp_path=Path(tmpdir), fit_params=self._fit_params)

    def _fit(
        self, train: DataTuple, tmp_path: Path, seed: int, model_dir: Optional[Path] = None
    ) -> FitParams:
        model_path = (model_dir.resolve() if model_dir is not None else tmp_path) / "model.npy"
        label_converter = LabelBinarizer()
        train_path = tmp_path / "train.json"
        self._create_file_in_zafar_format(train, train_path, label_converter)

        cmd = self._get_fit_cmd(str(train_path), str(model_path))
        working_dir = self._code_path.resolve() / self._sub_dir
        self.call_script(cmd, cwd=working_dir)

        return FitParams(model_path, label_converter)

    def _predict(self, test: TestTuple, tmp_path: Path, fit_params: FitParams) -> Prediction:
        test_path = tmp_path / "test.json"
        self._create_file_in_zafar_format(test, test_path, fit_params.label_converter)
        predictions_path = tmp_path / "predictions.json"
        cmd = self._get_predict_cmd(
            str(test_path), str(fit_params.model_path), str(predictions_path)
        )
        working_dir = self._code_path.resolve() / self._sub_dir
        self.call_script(cmd, cwd=working_dir)
        predictions = predictions_path.open().read()
        predictions = json.loads(predictions)

        predictions_correct = pd.Series([0 if x == -1 else 1 for x in predictions])
        return Prediction(hard=fit_params.label_converter.post_only_labels(predictions_correct))

    @abstractmethod
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        pass

    def _get_predict_cmd(self, test_name: str, model_path: str, output_file: str) -> List[str]:
        return ["predict.py", test_name, model_path, output_file]


class ZafarBaseline(_ZafarAlgorithmBase):
    """Zafar without fairness."""

    is_fairness_algo: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__(name="ZafarBaseline", sub_dir=SUB_DIR_IMPACT)

    @implements(InAlgorithm)
    def get_hyperparameters(self) -> HyperParamType:
        return {}

    @implements(_ZafarAlgorithmBase)
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "baseline", "0"]


class ZafarAccuracy(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, *, gamma: float = 0.5):
        super().__init__(name=f"ZafarAccuracy, γ={gamma}", sub_dir=SUB_DIR_IMPACT)
        self.gamma = gamma

    @implements(InAlgorithm)
    def get_hyperparameters(self) -> HyperParamType:
        return {"gamma": self.gamma}

    @implements(_ZafarAlgorithmBase)
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "gamma", str(self.gamma)]


class ZafarFairness(_ZafarAlgorithmBase):
    """Zafar with fairness."""

    def __init__(self, *, C: float = 0.001):
        super().__init__(name=f"ZafarFairness, C={C}", sub_dir=SUB_DIR_IMPACT)
        self._c = C

    @implements(InAlgorithm)
    def get_hyperparameters(self) -> HyperParamType:
        return {"C": self._c}

    @implements(_ZafarAlgorithmBase)
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return ["fit.py", train_name, model_path, "c", str(self._c)]


class ZafarEqOpp(_ZafarAlgorithmBase):
    """Zafar for Equality of Opportunity."""

    _mode: ClassVar[str] = "fnr"  # class level constant
    _base_name: ClassVar[str] = "ZafarEqOpp"

    def __init__(self, *, tau: float = 5.0, mu: float = 1.2, eps: float = 0.0001):
        name = f"{self._base_name}, τ={tau}, μ={mu} ε={eps}"
        super().__init__(name=name, sub_dir=SUB_DIR_MISTREAT)
        self._tau = tau
        self._mu = mu
        self._eps = eps

    @implements(InAlgorithm)
    def get_hyperparameters(self) -> HyperParamType:
        return {"tau": self._tau, "mu": self._mu, "eps": self._eps}

    @implements(_ZafarAlgorithmBase)
    def _get_fit_cmd(self, train_name: str, model_path: str) -> List[str]:
        return [
            "fit.py",
            train_name,
            model_path,
            self._mode,
            str(self._tau),
            str(self._mu),
            str(self._eps),
        ]


class ZafarEqOdds(ZafarEqOpp):
    """Zafar for Equalised Odds."""

    _mode: ClassVar[str] = "fprfnr"
    _base_name: ClassVar[str] = "ZafarEqOdds"
