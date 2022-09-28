"""Runs given metrics on the given results."""
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from ethicml.metrics.per_sensitive_attribute import MetricNotApplicable, PerSens, metric_per_sens
from ethicml.utility.data_structures import EvalTuple, Prediction

if TYPE_CHECKING:  # the following imports are only needed for type checking
    from collections.abc import Set as AbstractSet

    from ethicml.metrics.metric import Metric


__all__ = ["run_metrics", "per_sens_metrics_check"]


def run_metrics(
    predictions: Prediction,
    actual: EvalTuple,
    metrics: Sequence[Metric] = (),
    per_sens_metrics: Sequence[Metric] = (),
    aggregation: PerSens | AbstractSet[PerSens] = PerSens.DIFFS_RATIOS,
    use_sens_name: bool = True,
) -> dict[str, float]:
    """Run all the given metrics on the given predictions and return the results.

    :param predictions: DataFrame with predictions
    :param actual: EvalTuple with the labels
    :param metrics: list of metrics (Default: ())
    :param per_sens_metrics: list of metrics that are computed per sensitive attribute (Default: ())
    :param aggregation: Optionally specify aggregations that are performed on the per-sens metrics.
        (Default: ``DIFFS_RATIOS``)
    :param use_sens_name: if True, use the name of the senisitive variable in the returned results.
        If False, refer to the sensitive variable as "S". (Default: ``True``)
    :returns: A dictionary of all the metric results.
    """
    result: dict[str, float] = {}
    if predictions.hard.isna().any(axis=None):  # type: ignore[arg-type]
        return {"algorithm_failed": 1.0}
    for metric in metrics:
        result[metric.name] = metric.score(predictions, actual)

    for metric in per_sens_metrics:
        per_sens = metric_per_sens(predictions, actual, metric, use_sens_name)
        agg_funcs: AbstractSet[PerSens] = (
            {aggregation} if isinstance(aggregation, PerSens) else aggregation
        )
        # we can't add the aggregations directly to ``per_sens`` because then
        # we would create aggregations of aggregations
        aggregations: dict[str, float] = {}
        for agg in agg_funcs:
            aggregations.update(agg.func(per_sens))
        per_sens.update(aggregations)
        for key, value in per_sens.items():
            result[f"{metric.name}_{key}"] = value
    return result  # SUGGESTION: we could return a DataFrame here instead of a dictionary


def per_sens_metrics_check(per_sens_metrics: Sequence[Metric]) -> None:
    """Check if the given metrics allow application per sensitive attribute."""
    for metric in per_sens_metrics:
        if not metric.apply_per_sensitive:
            raise MetricNotApplicable(
                f"Metric {metric.name} is not applicable per sensitive "
                f"attribute, apply to whole dataset instead"
            )
