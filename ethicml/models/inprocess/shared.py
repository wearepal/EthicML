"""Methods that are shared among the inprocess algorithms."""
from __future__ import annotations
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
        if classifier is ClassifierType.lr:
            C = LogisticRegression().C  # type: ignore[attr-defined]
        elif classifier is ClassifierType.svm:
            C = SVC().C  # type: ignore[attr-defined]
        else:
            raise NotImplementedError(f'Unsupported classifier "{classifier}".')

    if kernel is None:
        kernel = (
            KernelType[SVC().kernel]  # type: ignore[attr-defined]
            if classifier is ClassifierType.svm
            else None
        )
    return C, kernel
