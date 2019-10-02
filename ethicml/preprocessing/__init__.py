"""
This moodule contains algorithms that pre-process the data in some way
"""

__all__ = [
    'LabelBinarizer',
    'get_biased_subset',
    'get_biased_and_debiased_subsets',
    'dataset_from_cond',
    'domain_split',
    'query_dt',
    'bin_cont_feats',
    'train_test_split',
]

from .adjust_labels import LabelBinarizer
from .biased_split import get_biased_subset, get_biased_and_debiased_subsets
from .domain_adaptation import dataset_from_cond, domain_split, query_dt
from .feature_binning import bin_cont_feats
from .train_test_split import train_test_split
