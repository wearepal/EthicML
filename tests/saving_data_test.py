"""
Test the saving data capability
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np

from ethicml.utility import DataTuple, TestTuple
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
        name='test data'
    )

    class CheckEquality(InAlgorithmAsync):
        """Dummy algorithm class for testing whether writing and reading feather files works"""

        def name(self):
            return "Check equality"

        def _script_command(self, train_paths, _, pred_path):
            """Check if the dataframes loaded from the files are the same as the original ones"""
            loaded = train_paths.load_from_feather()
            pd.testing.assert_frame_equal(data_tuple.x, loaded.x)
            pd.testing.assert_frame_equal(data_tuple.s, loaded.s)
            pd.testing.assert_frame_equal(data_tuple.y, loaded.y)
            # the following command copies the x of the training data to the pred_path location
            return ['-c', f'import shutil; shutil.copy("{train_paths.x}", "{pred_path}")']

    data_x = run_blocking(CheckEquality().run_async(data_tuple, data_tuple))
    pd.testing.assert_frame_equal(data_tuple.x, data_x)


def test_dataset_name_none():
    datatup = DataTuple(
        x=pd.DataFrame([3.0], columns=['a1']),
        s=pd.DataFrame([4.0], columns=['b2']),
        y=pd.DataFrame([6.0], columns=['c3']),
        name=None,
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        paths = datatup.write_as_feather(tmp_path, "pytest")
        # reload from feather file
        reloaded = paths.load_from_feather()
    assert reloaded.name is None
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_frame_equal(datatup.s, reloaded.s)
    pd.testing.assert_frame_equal(datatup.y, reloaded.y)


def test_dataset_name_with_spaces():
    name = "This is a very@#$%^&*((())) complicated name"
    datatup = TestTuple(
        x=pd.DataFrame([3.0], columns=['a1']),
        s=pd.DataFrame([4.0], columns=['b2']),
        name=name,
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        paths = datatup.write_as_feather(tmp_path, "pytest2")
        # reload from feather file
        reloaded = paths.load_from_feather()
    assert name == reloaded.name
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_frame_equal(datatup.s, reloaded.s)


def test_apply_to_joined_df():
    datatup = DataTuple(
        x=pd.DataFrame([3.0], columns=['a1']),
        s=pd.DataFrame([4.0], columns=['b2']),
        y=pd.DataFrame([6.0], columns=['c3']),
        name=None,
    )

    def _identity(x: pd.DataFrame):
        return x

    result = datatup.apply_to_joined_df(_identity)
    pd.testing.assert_frame_equal(datatup.x, result.x)
    pd.testing.assert_frame_equal(datatup.s, result.s)
    pd.testing.assert_frame_equal(datatup.y, result.y)
