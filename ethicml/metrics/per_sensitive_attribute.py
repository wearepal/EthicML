"""Evaluator for a metric per sensitive attribute class."""
from __future__ import annotations
from enum import Flag, auto
from typing import Callable, Mapping

import pandas as pd

from ethicml.utility.data_structures import EvalTuple, Prediction, SoftPrediction

from .metric import Metric

__all__ = [
    "MetricNotApplicable",
    "PerSens",
    "aggregate_over_sens",
    "diff_per_sens",
    "max_per_sens",
    "metric_per_sens",
    "min_per_sens",
    "ratio_per_sens",
]


class MetricNotApplicable(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead."""


class PerSens(Flag):
    """Aggregation methods for metrics that are computed per sensitive attributes.

    The members of this class are *flags*, which means that they can be combined
    with bit-wise *or*.

    :param value: Turn a value into a ``Flag`` member.

    :example:

    >>> PerSens.DIFFS | PerSens.MAX
    <PerSens.MAX|DIFFS: 3>
    >>> (PerSens.DIFFS | PerSens.RATIOS) == PerSens.DIFFS_RATIOS
    True
    >>> PerSens.DIFFS in PerSens.DIFFS_RATIOS
    True
    >>> PerSens.DIFFS in PerSens.DIFFS
    True
    >>> PerSens.DIFFS in PerSens.MIN_MAX
    False
    """

    DIFFS = auto()
    """Differences of the per-group results."""
    MAX = auto()
    """Maximum of the per-group results."""
    MIN = auto()
    """Minimum of the per-group results."""
    RATIOS = auto()
    """Ratios of the per-group results."""
    DIFFS_RATIOS = DIFFS | RATIOS
    """Differences and ratios of the per-group results."""
    MIN_MAX = MIN | MAX
    """Minimum and maximum of the per-group results."""
    ALL = DIFFS | RATIOS | MIN | MAX
    """All aggregations."""


def metric_per_sens(
    prediction: Prediction, actual: EvalTuple, metric: Metric, use_sens_name: bool = True
) -> dict[str, float]:
    """Compute a metric repeatedly on subsets of the data that share a senstitive attribute."""
    if not metric.apply_per_sensitive:
        raise MetricNotApplicable(
            f"Metric {metric.name} is not applicable per sensitive "
            f"attribute, apply to whole dataset instead"
        )

    assert actual.s.shape[0] == actual.y.shape[0]
    assert prediction.hard.shape[0] == actual.y.shape[0]

    per_sensitive_attr: dict[str, float] = {}

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


def aggregate_over_sens(
    per_sens_res: Mapping[str, float],
    aggregator: Callable[[float, float], float],
    infix: str,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    """Aggregate metrics over sensitive attributes.

    :param per_sens_res: Dictionary of the results.
    :param aggregator: A callable that is used to aggregate results.
    :returns: Dictionary of the aggregated results.
    """
    sens_keys = sorted(per_sens_res.keys())
    aggregated_over_sens: dict[str, float] = {}

    for i, sens_key_i in enumerate(sens_keys):
        i_value: float = per_sens_res[sens_key_i]
        for j in range(i + 1, len(sens_keys)):
            key: str = f"{prefix}{sens_key_i}{infix}{sens_keys[j]}{suffix}"
            j_value: float = per_sens_res[sens_keys[j]]

            aggregated_over_sens[key] = aggregator(i_value, j_value)

    return aggregated_over_sens


def diff_per_sens(per_sens_res: dict[str, float]) -> dict[str, float]:
    """Compute the difference in the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of differences
    """
    return aggregate_over_sens(per_sens_res, aggregator=_abs_diff, infix="-")


def _abs_diff(i_value: float, j_value: float) -> float:
    return abs(i_value - j_value)


def ratio_per_sens(per_sens_res: dict[str, float]) -> dict[str, float]:
    """Compute the ratios in the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of ratios
    """
    return aggregate_over_sens(per_sens_res, aggregator=_safe_ratio, infix="÷")


def _safe_ratio(i_value: float, j_value: float) -> float:
    min_val = min(i_value, j_value)
    max_val = max(i_value, j_value)

    return min_val / max_val if max_val != 0 else float("nan")


def min_per_sens(per_sens_res: dict[str, float]) -> dict[str, float]:
    """Compute the minimum value of the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of min values
    """
    return aggregate_over_sens(per_sens_res, aggregator=min, prefix="min(", infix=",", suffix=")")


def max_per_sens(per_sens_res: dict[str, float]) -> dict[str, float]:
    """Compute the maximum value of the metrics per sensitive attribute.

    :param per_sens_res: dictionary of the results
    :returns: dictionary of max values
    """
    return aggregate_over_sens(per_sens_res, aggregator=max, prefix="max(", infix=",", suffix=")")
