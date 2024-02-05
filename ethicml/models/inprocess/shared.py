"""Methods that are shared among the inprocess algorithms."""

from typing import Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from ethicml.utility import ClassifierType, KernelType

__all__ = ["LinearModel", "settings_for_svm_lr"]


LinearModel = Union[SVC, LinearSVC, LogisticRegression]


def settings_for_svm_lr(
    classifier: ClassifierType, C: float | None, kernel: KernelType | None
) -> tuple[float, KernelType | None]:
    """If necessary get the default settings for the C and kernel parameter of SVM and LR."""
    if classifier is ClassifierType.gbt:
        return 1.0, None
    if C is None:
        match classifier:
            case ClassifierType.lr:
                C = LogisticRegression().C  # type: ignore[attr-defined]
            case ClassifierType.svm:
                C = SVC().C  # type: ignore[attr-defined]
            case _:
                raise NotImplementedError(f'Unsupported classifier "{classifier}".')

    if kernel is None:
        kernel = KernelType[SVC().kernel] if classifier is ClassifierType.svm else None  # type: ignore[attr-defined]
    return C, kernel
