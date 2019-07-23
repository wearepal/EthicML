"""
This module contains items related to data, such as raw csv's and data objects
"""

from .adult import Adult
from .compas import Compas
from .configurable_dataset import ConfigurableDataset
from .credit import Credit
from .dataset import filter_features_by_prefixes, Dataset
from .german import German
from .load import load_data, create_data_obj
from .non_binary_toy import NonBinaryToy
from .sqf import Sqf
from .toy import Toy
