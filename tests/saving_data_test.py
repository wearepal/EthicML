"""
Test the saving data capability
"""
import pandas as pd
import numpy as np

from ethicml.utility import DataTuple
from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import InAlgorithmAsync


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
             'c3': np.array([0, 1, 0])}),
        name='test_data'
    )

    class CheckEquality(InAlgorithmAsync):
        """Dummy algorithm class for testing whether writing and reading feather files works"""

        def name(self):
            return "Check equality"

        def _script_command(self, train_paths, _, pred_path):
            """Check if the dataframes loaded from the files are the same as the original ones"""
            x_loaded = pd.read_feather(train_paths.x)
            s_loaded = pd.read_feather(train_paths.s)
            y_loaded = pd.read_feather(train_paths.y)
            pd.testing.assert_frame_equal(data_tuple.x, x_loaded)
            pd.testing.assert_frame_equal(data_tuple.s, s_loaded)
            pd.testing.assert_frame_equal(data_tuple.y, y_loaded)
            # the following command copies the x of the training data to the pred_path location
            return ['-c', f'import shutil; shutil.copy("{train_paths.x}", "{pred_path}")']

    data_x = run_blocking(CheckEquality().run_async(data_tuple, data_tuple))
    pd.testing.assert_frame_equal(data_tuple.x, data_x)
