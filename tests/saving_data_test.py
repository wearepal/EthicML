"""
Test the saving data capability
"""
import pandas as pd
import numpy as np

from ethicml.evaluators.evaluate_models import run_as_threaded
from ethicml.algorithms.utils import DataTuple
from ethicml.algorithms.inprocess.threaded.threaded_in_algorithm import ThreadedInAlgorithm


def test_simple_saving():
    data_tuple = DataTuple(
        x=pd.DataFrame(
            {'a1': np.array([3.2, 9.4, np.nan, 0.0]), 'a2': np.array([1, 1, 0, 1])}),
        s=pd.DataFrame(
            {'b1': np.array([1.8, -0.3, 1e10]),
             'b2': np.array([1, 1, -1]),
             'b3': np.array([0, 1, 0])}),
        y=pd.DataFrame(
            {'c1': np.array([-2, -3, np.nan]),
             'c3': np.array([0, 1, 0])})
    )

    class CheckEquality(ThreadedInAlgorithm):
        def run(self, train_paths, _, __):
            """Check if the dataframes loaded from the files are the same as the original ones"""
            x_loaded = pd.read_parquet(str(train_paths.x))
            s_loaded = pd.read_parquet(str(train_paths.s))
            y_loaded = pd.read_parquet(str(train_paths.y))
            pd.testing.assert_frame_equal(data_tuple.x, x_loaded)
            pd.testing.assert_frame_equal(data_tuple.s, s_loaded)
            pd.testing.assert_frame_equal(data_tuple.y, y_loaded)

    run_as_threaded(CheckEquality("Check equality"), data_tuple, data_tuple)
