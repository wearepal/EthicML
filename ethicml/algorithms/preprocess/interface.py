"""Methods that define commandline interfaces."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Mapping

if TYPE_CHECKING:
    from .pre_algorithm import PreAlgoArgs


def flag_interface(pre_algo_args: PreAlgoArgs, flags: Mapping[str, Any]) -> List[str]:
    """Generate the commandline arguments that are expected by the script about to be called.

    The flag interface consists of two strings, both JSON strings: the general pre-algo flags
    and then the more specific flags for the algorithm.
    """
    return [
        json.dumps(pre_algo_args, separators=(',', ':')),
        json.dumps(flags, separators=(',', ':')),
    ]
