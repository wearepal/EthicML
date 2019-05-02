"""
Set of scripts for splitting the train and test datasets based on conditions
"""

from typing import Tuple, Callable, Union, List
import pandas as pd
from ethicml.algorithms.utils import DataTuple


def lambda_factory(cond: str, val: str) -> Callable[[pd.Series], pd.Series]:
    """
    Given a condition and a value, return a lambda expression that can be used
    to get the indices of valid rows in a dataframe
    Args:
        cond: str
        val: str

    Returns: lamda

    """
    if cond == "==":
        def lambda_to_return(s):
            return s == float(val)
    elif cond == '&':
        def lambda_to_return(s):
            return s & float(val)
    else:
        raise NotImplementedError(f"We've not considered {cond}. Raise an issue and we'll fix it.")
    return lambda_to_return


def index_builder(comms: List[Union[pd.Series, str]]) -> pd.Series:
    """
    given a list of indices and operations on how to connect them,
    evaluate the operations left to right
    Args:
        comms:

    Returns:

    """
    conds = [comms[i:i + 2] for i in range(0, len(comms), 2)]

    first_op = conds.pop(0)
    if len(first_op) == 2:
        idxs: pd.Series = first_op[0]
        operation: str = first_op[1]
    else:
        return first_op[0]

    for cond in conds:
        if operation == '&':
            idxs = idxs & cond[0]
            try:
                operation = cond[1]
            except IndexError:
                pass
        elif operation == '|':
            idxs = idxs | cond[0]
            try:
                operation = cond[1]
            except IndexError:
                pass
        else:
            raise NotImplementedError(f"We've not considered {cond}. "
                                      f"Raise an issue and we'll fix it.")
    return idxs


def dataset_from_cond(dataset, cond):
    """
    return the datafram that meets some condition
    Args:
        dataset: DataFrame
        cond: condition

    Returns:

    """
    cond = cond.split()
    cond = [cond[i:i + 4] for i in range(0, len(cond), 4)]

    command = []
    for group in cond:
        bool_exp = None
        if len(group) == 4:
            bool_exp = group[3]

        lam = lambda_factory(group[1], group[2])

        command.append((dataset[group[0]][lam].index))
        if bool_exp:
            command.append(bool_exp)

    idxs = index_builder(command)

    return dataset.loc[idxs]


def domain_split(datatup: DataTuple, tr_cond: str, te_cond: str) -> Tuple[DataTuple, DataTuple]:
    """
    Splits a datatuple based on a condition
    Args:
        datatup: DataTuple
        tr_cond: condition for the training set
        te_cond: condition for the test set

    Returns: Tuple of DataTuple split into train and test. The test is all those that meet
    the test condition plus the same percentage again of the train set.

    """
    dataset = datatup.x

    train_dataset = dataset_from_cond(dataset, tr_cond)
    test_dataset = dataset_from_cond(dataset, te_cond)

    assert train_dataset.shape[0] + test_dataset.shape[0] == dataset.shape[0]

    test_pct = test_dataset.shape[0] / dataset.shape[0]
    train_pct = 1 - test_pct

    train_train_pcnt = (1 - (test_pct * 2)) / train_pct

    train_train = train_dataset.sample(frac=train_train_pcnt, random_state=888)
    test_train = train_dataset.drop(train_train.index)

    test = pd.concat([test_train, test_dataset], axis=0)

    train_x = datatup.x.loc[train_train.index].reset_index(drop=True)
    train_s = datatup.s.loc[train_train.index].reset_index(drop=True)
    train_y = datatup.y.loc[train_train.index].reset_index(drop=True)

    train_datatup = DataTuple(x=train_x, s=train_s, y=train_y)

    test_x = datatup.x.loc[test.index].reset_index(drop=True)
    test_s = datatup.s.loc[test.index].reset_index(drop=True)
    test_y = datatup.y.loc[test.index].reset_index(drop=True)

    test_datatup = DataTuple(x=test_x, s=test_s, y=test_y)

    return train_datatup, test_datatup
