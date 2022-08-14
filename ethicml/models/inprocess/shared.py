"""Methods that are shared among the inprocess algorithms."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ethicml.utility import ClassifierType, KernelType

__all__ = ["settings_for_svm_lr"]


def settings_for_svm_lr(
    classifier: ClassifierType, C: float | None, kernel: KernelType | None
) -> tuple[float, KernelType | None]:
    """If necessary get the default settings for the C and kernel parameter of SVM and LR."""
    if classifier is ClassifierType.gbt:
        return 1.0, None
    if C is None:
        if classifier is ClassifierType.lr:
            C = LogisticRegression().C
        elif classifier is ClassifierType.svm:
            C = SVC().C
        else:
            raise NotImplementedError(f'Unsupported classifier "{classifier}".')

    if kernel is None:
        kernel = KernelType[SVC().kernel] if classifier is ClassifierType.svm else None
    return C, kernel
