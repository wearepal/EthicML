"""Methods that are shared among the inprocess algorithms."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if TYPE_CHECKING:
    from .in_algorithm import InAlgoArgs

__all__ = ["flag_interface", "settings_for_svm_lr"]


def flag_interface(in_algo_args: InAlgoArgs, flags: Mapping[str, Any]) -> List[str]:
    """Generate the commandline arguments that are expected by the script about to be called.

    The flag interface consists of two strings, both JSON strings: the general in-algo flags
    and then the more specific flags for the algorithm.
    """
    return [
        json.dumps(in_algo_args, separators=(',', ':')),
        json.dumps(flags, separators=(',', ':')),
    ]


def settings_for_svm_lr(
    classifier: str, C: Optional[float], kernel: Optional[str]
) -> Tuple[float, str]:
    """If necessary get the default settings for the C and kernel parameter of SVM and LR."""
    if C is None:
        if classifier == "LR":
            C = LogisticRegression().C
        elif classifier == "SVM":
            C = SVC().C
        else:
            raise NotImplementedError(f'Unsupported classifier "{classifier}".')

    if kernel is None:
        kernel = SVC().kernel if classifier == "SVM" else ""
    return C, kernel
