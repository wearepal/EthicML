"""
Implementation of Heaviside decision function
"""
import pandas as pd

from ethicml.utility.data_structures import Predictions


def threshold(predictions: Predictions, threshold_at: float) -> Predictions:
    """
    thresholds soft ouputs so hard is 1 if soft >= threshold_at value, else 0
    Args:
        predictions:
        threshold_at:

    Returns:

    """

    def _thresh(x):
        return 1 if x >= threshold_at else 0

    labels = pd.DataFrame([_thresh(i) for i in predictions.soft['preds']], columns=['preds'])

    return Predictions(soft=predictions.soft, hard=labels)
