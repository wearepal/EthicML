"""Run Facct model."""
from pathlib import Path
from typing import Dict, List, Union

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithmAsync
from ethicml.algorithms.inprocess.shared import flag_interface


class Laftr(InAlgorithmAsync):
    """Madras et al. LAFTR model."""

    def __init__(
        self,
        dataset: str,
        *,
        batch_size: int = 64,
        device: int = -1,
        enc_additional_adv_steps: int = 0,
        enc_adv_weight: float = 1.0,
        enc_blocks: int = 2,
        enc_epochs: int = 30,
        enc_hidden_multiplier: int = 5,
        enc_ld: int = 50,
        enc_pred_weight: float = 1.0,
        enc_reg_weight: float = 1e-3,
        lr: float = 1e-3,
        seed: int = 0,
        warmup_steps: int = 0,
        weight_decay: float = 1e-6,
    ) -> None:
        """Laftr Algo."""
        super().__init__(name=f"Laftr Supervised")
        self.flags: Dict[str, Union[int, float]] = {
            "batch_size": batch_size,
            "dataset": dataset,
            "device": device,
            "enc_additional_adv_steps": enc_additional_adv_steps,
            "enc_adv_weight": enc_adv_weight,
            "enc_blocks": enc_blocks,
            "enc_epochs": enc_epochs,
            "enc_hidden_multiplier": enc_hidden_multiplier,
            "enc_ld": enc_ld,
            "enc_pred_weight": enc_pred_weight,
            "enc_reg_weight": enc_reg_weight,
            "lr": lr,
            "seed": seed,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
        }

    def _script_command(self, train_path: Path, test_path: Path, pred_path: Path) -> List[str]:
        args = flag_interface(train_path, test_path, pred_path, self.flags)
        return ["-m", "ethicml.implementations.laftr_impl"] + args
