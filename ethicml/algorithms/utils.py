"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a reasonable time
"""

from typing import Dict
import pandas as pd

def get_subset(train: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {
        'x': train['x'][:][:500],
        's': train['s'][:][:500],
        'y': train['y'][:][:500]
    }
