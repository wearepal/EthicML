"""Run Facct model."""
from pathlib import Path
from typing import Dict, List

from ethicml import InAlgorithmAsync
from ethicml.algorithms.inprocess.shared import flag_interface


class Facct(InAlgorithmAsync):
    """For FAccT."""

    def __init__(
        self,
        clf_epochs: int,
        enc_epochs: int,
        batch_size: int,
        enc_ld: int,
        pred_ld: int,
        warmup_steps: int = 0,
        wandb: int = 1,
        device: int = -1,
    ) -> None:
        """Facct Algo."""
        super().__init__(name=f"Facct")
        self.flags: Dict[str, int] = {
            "batch_size": batch_size,
            "clf_epochs": clf_epochs,
            "enc_epochs": enc_epochs,
            "enc_ld": enc_ld,
            "pred_ld": pred_ld,
            "wandb": wandb,
            "warmup_steps": warmup_steps,
            "device": device,
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.facct"] + args
