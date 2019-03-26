"""
Useful functions used in implementations
"""

from ethicml.algorithms.utils import DataTuple


def instance_weight_check(dataset: DataTuple):
    """
    Checks if there's an 'instance weight' field in the training features.
    If so, separate it out
    Args:
        dataset:

    Returns:

    """
    i_w = None
    if 'instance weights' in dataset.x.columns:
        i_w = dataset.x['instance weights']
        dataset = DataTuple(x=dataset.x.drop(['instance weights'], axis=1),
                            s=dataset.s,
                            y=dataset.y)
    return dataset, i_w
