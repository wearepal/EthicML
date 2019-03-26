"""
Implementatioon of Kamiran and Calders 2012

Heavily based on AIF360
https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/reweighing.py

    References:
        .. [4] F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.
"""
import argparse
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from ethicml.algorithms.algorithm_base import load_dataframe
from ethicml.algorithms.preprocess import PreAlgorithm
from ethicml.algorithms.utils import DataTuple


class Kamiran(PreAlgorithm):

    """
    Kamiran and Calders 2012
    """

    def __init__(self):
        # convert all parameter values to lists of strings
        flags: Dict[None, None] = {}
        super().__init__(flags)

        np.random.seed(888)

    @staticmethod
    def _obtain_conditionings(dataset: DataTuple):
        """Obtain the necessary conditioning boolean vectors to compute
        instance level weights.
        """
        y_col = dataset.y.columns[0]
        y_pos = dataset.y[y_col].max()
        y_neg = dataset.y[y_col].min()
        s_col = dataset.s.columns[0]
        s_pos = dataset.s[s_col].max()
        s_neg = dataset.s[s_col].min()

        # combination of label and privileged/unpriv. groups
        cond_p_fav = dataset.x[(dataset.y[y_col] == y_pos) & (dataset.s[s_col] == s_pos)]
        cond_p_unfav = dataset.x[(dataset.y[y_col] == y_neg) & (dataset.s[s_col] == s_pos)]
        cond_up_fav = dataset.x[(dataset.y[y_col] == y_pos) & (dataset.s[s_col] == s_neg)]
        cond_up_unfav = dataset.x[(dataset.y[y_col] == y_neg) & (dataset.s[s_col] == s_neg)]

        return cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav

    def _run(self, train: DataTuple, test: DataTuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
        (cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) = self._obtain_conditionings(train)

        y_col = train.y.columns[0]
        y_pos = train.y[y_col].max()
        y_neg = train.y[y_col].min()
        s_col = train.s.columns[0]
        s_pos = train.s[s_col].max()
        s_neg = train.s[s_col].min()

        num_samples = train.x.shape[0]
        n_p = train.s[train.s[s_col] == s_pos].shape[0]
        n_up = train.s[train.s[s_col] == s_neg].shape[0]
        n_fav = train.y[train.y[y_col] == y_pos].shape[0]
        n_unfav = train.y[train.y[y_col] == y_neg].shape[0]

        n_p_fav = cond_p_fav.shape[0]
        n_p_unfav = cond_p_unfav.shape[0]
        n_up_fav = cond_up_fav.shape[0]
        n_up_unfav = cond_up_unfav.shape[0]

        w_p_fav = n_fav * n_p / (num_samples * n_p_fav)
        w_p_unfav = n_unfav * n_p / (num_samples * n_p_unfav)
        w_up_fav = n_fav * n_up / (num_samples * n_up_fav)
        w_up_unfav = n_unfav * n_up / (num_samples * n_up_unfav)

        train_instance_weights = pd.DataFrame(1, index=np.arange(train.x.shape[0]),
                                              columns=["instance weights"])

        train_instance_weights.iloc[cond_p_fav.index] *= w_p_fav
        train_instance_weights.iloc[cond_p_unfav.index] *= w_p_unfav
        train_instance_weights.iloc[cond_up_fav.index] *= w_up_fav
        train_instance_weights.iloc[cond_up_unfav.index] *= w_up_unfav

        train_x = pd.concat((train.x, train_instance_weights), axis=1)

        return train_x, test.x

    @property
    def name(self) -> str:
        return "Kamiran & Calders"


def load_data(flags):
    """Load data from the paths specified in the flags"""
    train = DataTuple(
        x=load_dataframe(Path(flags.train_x)),
        s=load_dataframe(Path(flags.train_s)),
        y=load_dataframe(Path(flags.train_y)),
    )
    test = DataTuple(
        x=load_dataframe(Path(flags.test_x)),
        s=load_dataframe(Path(flags.test_s)),
        y=load_dataframe(Path(flags.test_y)),
    )
    return train, test


def _parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser()

    # paths to the files with the data
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_s", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_s", required=True)
    parser.add_argument("--test_y", required=True)

    # paths to where the processed inputs should be stored
    parser.add_argument("--train_new", required=True)
    parser.add_argument("--test_new", required=True)

    return parser.parse_args()


def main():
    """main method to run model"""
    flags = _parse_arguments()

    model = Kamiran()
    train, test = load_data(flags)
    model.save_transformations(model.run(train, test), flags)


if __name__ == "__main__":
    main()
