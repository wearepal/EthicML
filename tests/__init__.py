"""EthicML tests."""

from . import metrics
from .data import loading_data_test
from .metrics import nonparamaterized_metric_test

__all__ = ["loading_data_test", "metrics", "nonparamaterized_metric_test"]
