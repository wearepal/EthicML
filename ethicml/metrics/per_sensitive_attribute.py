"""Evaluator for a metric per sensitive attribute class."""

from typing import Dict

import pandas as pd

from ethicml.utility import DataTuple, EvalTuple, Prediction, SoftPrediction

from .metric import Metric

__all__ = [
    "metric_per_sensitive_attribute",
    "diff_per_sensitive_attribute",
    "ratio_per_sensitive_attribute",
    "MetricNotApplicable",
]


class MetricNotApplicable(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead."""


def metric_per_sensitive_attribute(
    prediction: Prediction, actual: EvalTuple, metric: Metric, use_sens_name: bool = True
) -> Dict[str, float]:
    """Compute a metric repeatedly on subsets of the data that share a senstitive attribute.

    :param prediction:
    :param actual:
    :param metric:
    :param use_sens_name:  (Default: True)
    """
    if not metric.apply_per_sensitive:
        raise MetricNotApplicable(
            f"Metric {metric.name} is not applicable per sensitive "
            f"attribute, apply to whole dataset instead"
        )

    assert actual.s.shape[0] == actual.y.shape[0]
    assert prediction.hard.shape[0] == actual.y.shape[0]

    per_sensitive_attr: Dict[str, float] = {}

    s_column: str = actual.s_column

    for unique_s in actual.s.unique():
        mask: pd.Series = actual.s == unique_s
        subset = actual.get_s_subset(unique_s)
        pred_y: Prediction
        if isinstance(prediction, SoftPrediction):
            pred_y = SoftPrediction(soft=prediction.soft[mask], info=prediction.info)
        else:
            pred_y = Prediction(
                hard=prediction.hard.loc[mask].reset_index(drop=True), info=prediction.info
            )
        key = (s_column if use_sens_name else "S") + "_" + str(unique_s)
        per_sensitive_attr[key] = metric.score(pred_y, subset)

    return per_sensitive_attr


def diff_per_sensitive_attribute(per_sens_res: Dict[str, float]) -> Dict[str, float]:
    """Compute the difference in the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of differences
    """
    sens_values = sorted(per_sens_res.keys())
    diff_per_sens = {}

    for i, _ in enumerate(sens_values):
        i_value: float = per_sens_res[sens_values[i]]
        for j in range(i + 1, len(sens_values)):
            key: str = f"{sens_values[i]}-{sens_values[j]}"
            j_value: float = per_sens_res[sens_values[j]]
            diff_per_sens[key] = abs(i_value - j_value)

    return diff_per_sens


def ratio_per_sensitive_attribute(per_sens_res: Dict[str, float]) -> Dict[str, float]:
    """Compute the ratios in the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of ratios
    """
    sens_values = sorted(per_sens_res.keys())
    ratio_per_sens = {}

    for i, _ in enumerate(sens_values):
        i_value: float = per_sens_res[sens_values[i]]
        for j in range(i + 1, len(sens_values)):
            key: str = f"{sens_values[i]}/{sens_values[j]}"
            j_value: float = per_sens_res[sens_values[j]]

            min_val = min(i_value, j_value)
            max_val = max(i_value, j_value)

            ratio_per_sens[key] = min_val / max_val if max_val != 0 else float("nan")

    return ratio_per_sens
