from typing import List, Sequence, Tuple

from ethicml.algorithms.preprocess import PreAlgorithmAsync
from ethicml.algorithms.preprocess.interface import flag_interface
from ethicml.utility import PathTuple, TestPathTuple, DataTuple, TestTuple


class Imagine(PreAlgorithmAsync):
    def __init__(
        self,
        enc_size: Sequence[int] = (40,),
        adv_size: Sequence[int] = (40,),
        pred_size: Sequence[int] = (40,),
        batch_size: int = 64,
        epochs: int = 50,
        adv_weight: float = 1.0,
        validation_pcnt: float = 0.1,
        dataset: str = "Toy",
        sample: int = 1,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.enc_size = enc_size
        self.adv_size = adv_size
        self.pred_size = pred_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.adv_weight = adv_weight
        self.validation_pcnt = validation_pcnt
        self.dataset = dataset
        self.sample = sample

    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        from ...implementations import imagine  # only import this on demand because of pytorch

        # SUGGESTION: it would be great if ImagineSettings could already be created in the init
        flags = imagine.ImagineSettings(
            enc_size=self.enc_size,
            adv_size=self.adv_size,
            pred_size=self.pred_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            adv_weight=self.adv_weight,
            validation_pcnt=self.validation_pcnt,
            dataset=self.dataset,
            sample=self.sample,
        )
        return imagine.train_and_transform(train, test, flags)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        new_train_paths: PathTuple,
        new_test_paths: TestPathTuple,
    ) -> List[str]:
        args = flag_interface(train_paths, test_paths, new_train_paths, new_test_paths, vars(self))
        return ["-m", "ethicml.implementations.imagine"] + args

    @property
    def name(self) -> str:
        return "Imagined Examples"
