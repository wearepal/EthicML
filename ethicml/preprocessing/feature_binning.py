import pandas as pd
from itertools import groupby
from typing import List, Dict

from ethicml.utility import DataTuple


def bin_cont_feats(data: DataTuple) -> DataTuple:
    x_name: List[str] = data.x.columns
    groups: List[List[str]] = [list(group) for key, group in groupby(x_name, lambda x: x.split('_')[0])]

    copy = data.x.copy()

    for group in groups:
        if len(group) == 1 and int(data.x[group].nunique()) > 2:
            copy[group] = pd.cut(data.x[group].to_numpy()[:, 0], 5)
            copy = pd.concat([copy, pd.get_dummies(copy[group])], axis=1)
            copy = copy.drop(group, axis=1)

    return DataTuple(x=copy, s=data.s, y=data.y)
