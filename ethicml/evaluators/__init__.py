"""
This module contains evaluators which apply algorithms over datasets and obtain metrics
"""

from .cross_validator import CrossValidator, CVResults
from .evaluate_models import evaluate_models, run_metrics, load_results
from .parallelism import arrange_in_parallel, run_in_parallel
from .per_sensitive_attribute import (
    metric_per_sensitive_attribute,
    diff_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
    MetricNotApplicable,
)
