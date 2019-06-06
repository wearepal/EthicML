"""
This module contains evaluators which apply algorithms over datasets and obtain metrics
"""


from . import evaluate_models, per_sensitive_attribute
from .parallelism import arrange_in_parallel, run_in_parallel
