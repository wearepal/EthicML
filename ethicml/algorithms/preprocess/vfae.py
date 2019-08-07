"""
Variational Fair Auto-Encoder by Louizos et al
"""

from typing import List, Tuple, Dict, Union, Optional

from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithmAsync
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.utility.data_structures import PathTuple, TestPathTuple, DataTuple, TestTuple


class VFAE(PreAlgorithmAsync):
    """
    VFAE Object - see implementation file for details
    """

    def __init__(
        self,
        dataset: str,
        supervised: bool = True,
        epochs: int = 10,
        batch_size: int = 32,
        fairness: str = "DI",
        z1_enc_size: Optional[List[int]] = None,
        z2_enc_size: Optional[List[int]] = None,
        z1_dec_size: Optional[List[int]] = None,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

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
            "z1_enc_size": z1_enc_size,
            "z2_enc_size": z2_enc_size,
            "z1_dec_size": z1_dec_size,
        }

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        from ...implementations.vfae import train_and_transform

        test = TestTuple(x=test.x, s=test.s, name=test.name)

        return train_and_transform(train, test, self.flags)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, self.flags)
        return ["-m", "ethicml.implementations.vfae"] + args

    @property
    def name(self) -> str:
        return "VFAE"
