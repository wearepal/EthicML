"""Variational Fair Auto-Encoder by Louizos et al."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from ranzen import implements

from .interface import flag_interface
from .pre_algorithm import PreAlgorithmAsync

__all__ = ["VFAE"]


class VFAE(PreAlgorithmAsync):
    """VFAE Object - see implementation file for details."""

    def __init__(
        self,
        dir: Union[str, Path],
        dataset: str,
        *,
        supervised: bool = True,
        epochs: int = 10,
        batch_size: int = 32,
        fairness: str = "DI",
        latent_dims: int = 50,
        z1_enc_size: Optional[List[int]] = None,
        z2_enc_size: Optional[List[int]] = None,
        z1_dec_size: Optional[List[int]] = None,
        seed: int = 888,
    ):
        self.seed = seed
        self._out_size = latent_dims
        self.model_dir = dir if isinstance(dir, Path) else Path(dir)

        if z1_enc_size is None:
            z1_enc_size = [100]
        if z2_enc_size is None:
            z2_enc_size = [100]
        if z1_dec_size is None:
            z1_dec_size = [100]

        self.flags: Dict[str, Union[int, str, List[int]]] = {
            "supervised": supervised,
            "fairness": fairness,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset,
            "latent_dims": latent_dims,
            "z1_enc_size": z1_enc_size,
            "z2_enc_size": z2_enc_size,
            "z1_dec_size": z1_dec_size,
            "seed": seed,
        }

    @property
    def name(self) -> str:
        """Name of the algorithm."""
        return "VFAE"

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
        return ["-m", "ethicml.implementations.vfae"] + args

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
        return ["-m", "ethicml.implementations.vfae"] + args

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
        return ["-m", "ethicml.implementations.vfae"] + args
