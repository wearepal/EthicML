"""
Test the saving data capability
"""
import pandas as pd
import numpy as np

from ethicml.algorithms.utils import DataTuple
from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm


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

    class CheckEquality(InAlgorithm):
        """Dummy algorithm class for testing whether writing and reading parquet files works"""
        def _run(self, *_):
            pass

        def name(self):
            return "Check equality"

        def run_thread(self, train_paths, _, __):
            """Check if the dataframes loaded from the files are the same as the original ones"""
            x_loaded = pd.read_parquet(str(train_paths.x))
            s_loaded = pd.read_parquet(str(train_paths.s))
            y_loaded = pd.read_parquet(str(train_paths.y))
            pd.testing.assert_frame_equal(data_tuple.x, x_loaded)
            pd.testing.assert_frame_equal(data_tuple.s, s_loaded)
            pd.testing.assert_frame_equal(data_tuple.y, y_loaded)
            return train_paths.x

    data_x = CheckEquality().run_threaded(data_tuple, data_tuple)
    pd.testing.assert_frame_equal(data_tuple.x, data_x)
