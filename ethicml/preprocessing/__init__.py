"""
This moodule contains algorithms that pre-process the data in some way
"""

from .adjust_labels import LabelBinarizer
from .domain_adaptation import apply_to_joined_tuple, dataset_from_cond, domain_split, query_dt
from .train_test_split import train_test_split
