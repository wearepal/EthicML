"""
implementation of the upsampling method
"""

import pandas as pd
from ethicml.implementations.utils import (
    pre_algo_argparser,
    load_data_from_flags,
    save_transformations,
)
from ethicml.utility import DataTuple, TestTuple


def concat_datatuples(first_dt, second_dt):
    """
    Given 2 datatuples, concatenate them and shuffle
    Args:
        first_dt: DataTuple
        second_dt: DataTuple

    Returns: DataTuple

    """
    assert (first_dt.x.columns == second_dt.x.columns).all()
    assert (first_dt.s.columns == second_dt.s.columns).all()
    assert (first_dt.y.columns == second_dt.y.columns).all()

    x_columns: pd.Index = first_dt.x.columns
    s_columns: pd.Index = first_dt.s.columns
    y_columns: pd.Index = first_dt.y.columns

    a_combined: pd.DataFrame = pd.concat([first_dt.x, first_dt.s, first_dt.y], axis="columns")
    b_combined: pd.DataFrame = pd.concat([second_dt.x, second_dt.s, second_dt.y], axis="columns")

    combined: pd.DataFrame = pd.concat([a_combined, b_combined], axis='rows')
    combined = combined.sample(frac=1.0, random_state=1).reset_index(drop=True)

    return DataTuple(
        x=combined[x_columns], s=combined[s_columns], y=combined[y_columns], name=first_dt.name
    )


def upsample(dataset):
    """
    Upsample a datatuple
    Args:
        dataset:

    Returns:

    """
    s_col = dataset.s.columns[0]
    y_col = dataset.y.columns[0]

    s_vals = dataset.s[s_col].unique()
    y_vals = dataset.y[y_col].unique()

    import itertools

    groups = itertools.product(s_vals, y_vals)

    data = {}
    for s, y in groups:
        data[(s, y)] = DataTuple(
            x=dataset.x[(dataset.s[s_col] == s) & (dataset.y[y_col] == y)].reset_index(drop=True),
            s=dataset.s[(dataset.s[s_col] == s) & (dataset.y[y_col] == y)].reset_index(drop=True),
            y=dataset.y[(dataset.s[s_col] == s) & (dataset.y[y_col] == y)].reset_index(drop=True),
        )

    percentages = {}

    vals = []
    for key, val in data.items():
        vals.append(val.x.shape[0])

    for key, val in data.items():
        percentages[key] = max(vals) / val.x.shape[0]

    x_columns: pd.Index = dataset.x.columns
    s_columns: pd.Index = dataset.s.columns
    y_columns: pd.Index = dataset.y.columns

    upsampled = {}
    for key, val in data.items():
        all_data: pd.DataFrame = pd.concat([val.x, val.s, val.y], axis="columns")
        all_data = all_data.sample(frac=percentages[key], random_state=1, replace=True).reset_index(
            drop=True
        )
        upsampled[key] = DataTuple(
            x=all_data[x_columns], s=all_data[s_columns], y=all_data[y_columns]
        )

    upsampled_datatuple = None
    for key, val in upsampled.items():
        if upsampled_datatuple is None:
            upsampled_datatuple = val
        else:
            upsampled_datatuple = concat_datatuples(upsampled_datatuple, val)

    return upsampled_datatuple


def train_and_transform(train, test):
    """
    Tran and transform function for the upsampler method
    Args:
        train:
        test:

    Returns:

    """
    upsampled_train = upsample(train)

    return upsampled_train, TestTuple(x=test.x, s=test.s)


def main():
    """This function runs the SVM model as a standalone program"""
    parser = pre_algo_argparser()

    args = parser.parse_args()
    train, test = load_data_from_flags(vars(args))
    save_transformations(train_and_transform(train, test), args)


if __name__ == "__main__":
    main()
