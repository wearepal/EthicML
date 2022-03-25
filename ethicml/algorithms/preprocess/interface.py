"""Methods that define commandline interfaces."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .pre_algorithm import PreAlgoArgs


def flag_interface(flags: Mapping[str, Any], pre_algo_args: PreAlgoArgs) -> str:
    """Generate the commandline arguments that are expected by the script about to be called."""
    return json.dumps({**flags, **pre_algo_args}, separators=(',', ':'))
