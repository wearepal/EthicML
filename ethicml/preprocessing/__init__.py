"""
This moodule contains algorithms that pre-process the data in some way
"""

__all__ = [
    'BiasedDebiasedSubsets',
    'BiasedSubset',
    'DataSplitter',
    'LabelBinarizer',
    'ProportionalTrainTestSplit',
    'TrainTestSplit',
    'bin_cont_feats',
    'dataset_from_cond',
    'domain_split',
    'get_biased_and_debiased_subsets',
    'get_biased_subset',
    'query_dt',
    'train_test_split',
]

from .adjust_labels import LabelBinarizer
from .biased_split import (
    BiasedDebiasedSubsets,
    BiasedSubset,
    get_biased_and_debiased_subsets,
    get_biased_subset,
)
from .domain_adaptation import dataset_from_cond, domain_split, query_dt
from .feature_binning import bin_cont_feats
from .train_test_split import (
    DataSplitter,
    ProportionalTrainTestSplit,
    TrainTestSplit,
    train_test_split,
)
