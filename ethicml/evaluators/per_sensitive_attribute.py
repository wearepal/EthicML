"""
Evaluator for a metric per sensitive attribute class
"""

from typing import Dict, List
import pandas as pd

from ..metrics.metric import Metric
from ..algorithms.utils import DataTuple


class MetricNotApplicable(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead"""


def metric_per_sensitive_attribute(
        predictions: pd.DataFrame,
        actual: DataTuple,
        metric: Metric) -> Dict[str, float]:
    """Compute a metric repeatedly on subsets of the data that share a senstitive attribute"""
    if not metric.apply_per_sensitive:
        raise MetricNotApplicable(f"Metric {metric.name} is not applicable per sensitive "
                                  f"attribute, apply to whole dataset instead")

    amalgamated = pd.concat([actual.x,
                             actual.s,
                             actual.y,
                             predictions], axis=1)

    if amalgamated.shape[0] != actual.x.shape[0]:
        print("ddf")
    assert amalgamated.shape[0] == actual.x.shape[0]

    per_sensitive_attr: Dict[str, float] = {}

    s_columns: List[str] = [s_col for s_col in actual.s.columns]
    y_columns: List[str] = [y_col for y_col in actual.y.columns]
    pred_column: List[str] = [p_col for p_col in predictions.columns]
    assert len(y_columns) == 1

    for y_col in y_columns:
        for s_col in s_columns:
            for unique_s in actual.s[s_col].unique():
                for p_col in pred_column:
                    subset = DataTuple(
                        x=amalgamated[actual.s[s_col] == unique_s][actual.x.columns],
                        s=amalgamated[actual.s[s_col] == unique_s][s_col],
                        y=amalgamated[actual.s[s_col] == unique_s][y_col])
                    pred_y = amalgamated[actual.s[s_col] == unique_s][p_col]
                    key = s_col + '_' + str(unique_s)
                    per_sensitive_attr[key] = metric.score(pred_y, subset)

    return per_sensitive_attr


def diff_per_sensitive_attribute(per_sens_res: Dict[str, float]) -> Dict[str, float]:
    """

    Args:
        per_sens_res:

    Returns:

    """
    sens_values = list(per_sens_res.keys())
    sens_values.sort()
    diff_per_sens = {}

    for i, _ in enumerate(sens_values):
        for j in range(i+1, len(sens_values)):
            key: str = "{}-{}".format(sens_values[i], sens_values[j])
            i_value: float = per_sens_res[sens_values[i]]
            j_value: float = per_sens_res[sens_values[j]]
            diff_per_sens[key] = abs(i_value - j_value)

    return diff_per_sens


def ratio_per_sensitive_attribute(per_sens_res: Dict[str, float]) -> Dict[str, float]:
    """

    Args:
        per_sens_res:

    Returns:

    """
    sens_values = list(per_sens_res.keys())
    sens_values.sort()
    ratio_per_sens = {}

    for i, _ in enumerate(sens_values):
        for j in range(i+1, len(sens_values)):
            key: str = "{}/{}".format(sens_values[i], sens_values[j])
            i_value: float = per_sens_res[sens_values[i]]
            j_value: float = per_sens_res[sens_values[j]]
            ratio_per_sens[key] = i_value / j_value

    return ratio_per_sens
