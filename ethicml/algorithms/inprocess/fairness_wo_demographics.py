"""Fairness without Demographics."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from ranzen import implements

from .in_algorithm import InAlgorithmAsync
from .shared import flag_interface

__all__ = ["DRO"]


class DRO(InAlgorithmAsync):
    """Implementation of https://arxiv.org/abs/1806.08010 ."""

    def __init__(
        self,
        dir: Union[str, Path],
        eta: float = 0.5,
        epochs: int = 10,
        batch_size: int = 32,
        network_size: Optional[List[int]] = None,
        seed: int = 888,
    ):
        self.seed = seed
        self.is_fairness_algo = True
        if network_size is None:
            network_size = [50]
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)
        self.flags: Dict[str, Union[float, int, str, List[int]]] = {
            "eta": eta,
            "batch_size": batch_size,
            "epochs": epochs,
            "network_size": network_size,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "Dist Robust Optim"

    @implements(InAlgorithmAsync)
    def _run_script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(
            train_path=train_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.dro_tabular"] + args

    @implements(InAlgorithmAsync)
    def _fit_script_command(self, train_path: Path, model_path: Path) -> List[str]:
        args = flag_interface(train_path=train_path, model_path=model_path, flags=self.flags)
        return ["-m", "ethicml.implementations.dro_tabular"] + args

    @implements(InAlgorithmAsync)
    def _predict_script_command(
        self, model_path: Path, test_path: Path, pred_path: Path
    ) -> List[str]:
        args = flag_interface(
            model_path=model_path, test_path=test_path, pred_path=pred_path, flags=self.flags
        )
        return ["-m", "ethicml.implementations.dro_tabular"] + args
