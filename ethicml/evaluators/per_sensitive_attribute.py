"""
Evaluator for a metric per sensitive attribute class
"""

from typing import Dict, List
import pandas as pd

from ethicml.utility.data_structures import DataTuple
from ..metrics.metric import Metric


class MetricNotApplicable(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead"""


def metric_per_sensitive_attribute(
    predictions: pd.DataFrame, actual: DataTuple, metric: Metric
) -> Dict[str, float]:
    """Compute a metric repeatedly on subsets of the data that share a senstitive attribute"""
    if not metric.apply_per_sensitive:
        raise MetricNotApplicable(
            f"Metric {metric.name} is not applicable per sensitive "
            f"attribute, apply to whole dataset instead"
        )

    assert actual.s.shape[0] == actual.x.shape[0]
    assert actual.s.shape[0] == actual.y.shape[0]

    per_sensitive_attr: Dict[str, float] = {}

    s_columns: List[str] = [s_col for s_col in actual.s.columns]
    y_columns: List[str] = [y_col for y_col in actual.y.columns]
    pred_column: List[str] = [p_col for p_col in predictions.columns]
    assert len(y_columns) == 1

    for y_col in y_columns:
        for s_col in s_columns:
            for unique_s in actual.s[s_col].unique():
                for p_col in pred_column:
                    mask: pd.Series = (actual.s[s_col] == unique_s)
                    subset = DataTuple(
                        x=pd.DataFrame(
                            actual.x.loc[mask][actual.x.columns], columns=actual.x.columns
                        ).reset_index(drop=True),
                        s=pd.DataFrame(actual.s.loc[mask][s_col], columns=[s_col]).reset_index(
                            drop=True
                        ),
                        y=pd.DataFrame(actual.y.loc[mask][y_col], columns=[y_col]).reset_index(
                            drop=True
                        ),
                        name=actual.name,
                    )
                    pred_y = pd.DataFrame(
                        predictions.loc[mask][p_col], columns=[p_col]
                    ).reset_index(drop=True)
                    key = s_col + "_" + str(unique_s)
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
        for j in range(i + 1, len(sens_values)):
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
        for j in range(i + 1, len(sens_values)):
            key: str = "{}/{}".format(sens_values[i], sens_values[j])
            i_value: float = per_sens_res[sens_values[i]]
            j_value: float = per_sens_res[sens_values[j]]

            min_val = min(i_value, j_value)
            max_val = max(i_value, j_value)

            ratio_per_sens[key] = min_val / max_val

    return ratio_per_sens
