"""Zemel's Learned Fair Representations."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ranzen import implements

from .interface import flag_interface
from .pre_algorithm import PreAlgorithm, PreAlgorithmAsync

__all__ = ["Zemel"]

from ethicml.utility import DataTuple, TestTuple


class Zemel(PreAlgorithmAsync):
    """AIF360 implementation of Zemel's LFR."""

    def __init__(
        self,
        dir: Union[str, Path],
        threshold: float = 0.5,
        clusters: int = 2,
        Ax: float = 0.01,
        Ay: float = 0.1,
        Az: float = 0.5,
        max_iter: int = 5_000,
        maxfun: int = 5_000,
        epsilon: float = 1e-5,
        seed: int = 888,
    ) -> None:
        self.seed = seed
        self._out_size: Optional[int] = None
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        self.flags: Dict[str, Union[int, float]] = {
            "clusters": clusters,
            "Ax": Ax,
            "Ay": Ay,
            "Az": Az,
            "max_iter": max_iter,
            "maxfun": maxfun,
            "epsilon": epsilon,
            "threshold": threshold,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Zemel"

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        self._out_size = train.x.shape[1]
        return super().run(train, test)

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple) -> Tuple[PreAlgorithm, DataTuple]:
        self._out_size = train.x.shape[1]
        return super().fit(train)

    @implements(PreAlgorithmAsync)
    def _run_script_command(
        self, train_path: Path, test_path: Path, new_train_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(
            train_path=train_path,
            test_path=test_path,
            new_train_path=new_train_path,
            new_test_path=new_test_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.zemel"] + args

    @implements(PreAlgorithmAsync)
    def _fit_script_command(
        self, train_path: Path, new_train_path: Path, model_path: Path
    ) -> List[str]:
        args = flag_interface(
            train_path=train_path,
            new_train_path=new_train_path,
            model_path=model_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.zemel"] + args

    @implements(PreAlgorithmAsync)
    def _transform_script_command(
        self, model_path: Path, test_path: Path, new_test_path: Path
    ) -> List[str]:
        args = flag_interface(
            model_path=model_path,
            test_path=test_path,
            new_test_path=new_test_path,
            flags=self.flags,
        )
        return ["-m", "ethicml.implementations.zemel"] + args
