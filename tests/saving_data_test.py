"""Test the saving data capability."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from ethicml import DataTuple, InAlgorithmAsync, Prediction, TestTuple, run_blocking


def test_simple_saving() -> None:
    """Tests that a DataTuple can be saved."""
    data_tuple = DataTuple(
        x=pd.DataFrame({"a1": np.array([3.2, 9.4, np.nan, 0.0])}),
        s=pd.DataFrame(
            {
                "b1": np.array([18, -3, int(1e10)]),
                "b2": np.array([1, 1, -1]),
                "b3": np.array([0, 1, 0]),
            }
        ),
        y=pd.DataFrame({"c1": np.array([-2.0, -3.0, np.nan]), "c3": np.array([0.0, 1.0, 0.0])}),
        name="test data",
    )

    class CheckEquality(InAlgorithmAsync):
        """Dummy algorithm class for testing whether writing and reading feather files works."""

        def __init__(self) -> None:
            super().__init__(name="Check equality")

        def _script_command(self, train_path, _, pred_path):
            """Check if the dataframes loaded from the files are the same as the original ones."""
            loaded = DataTuple.from_npz(train_path)
            pd.testing.assert_frame_equal(data_tuple.x, loaded.x)
            pd.testing.assert_frame_equal(data_tuple.s, loaded.s)
            pd.testing.assert_frame_equal(data_tuple.y, loaded.y)
            # write a file for the predictions
            np.savez(pred_path, hard=np.load(train_path)["x"])
            return ["-c", "pass"]

    data_x = run_blocking(CheckEquality().run_async(data_tuple, data_tuple))
    pd.testing.assert_series_equal(data_tuple.x["a1"], data_x.hard, check_names=False)


def test_predictions_loaded(temp_dir) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]))
    preds.to_npz(temp_dir / "test.npz")
    loaded = Prediction.from_npz(temp_dir / "test.npz")
    pd.testing.assert_series_equal(preds.hard, loaded.hard, check_dtype=False)


def test_predictions_info_loaded(temp_dir) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]), info={"sample": 123.4})
    preds.to_npz(temp_dir / "test.npz")
    loaded = Prediction.from_npz(temp_dir / "test.npz")
    pd.testing.assert_series_equal(preds.hard, loaded.hard, check_dtype=False)
    assert preds.info == loaded.info


def test_predictions_info_loaded_bad(temp_dir) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]), info={"sample": np.array([1, 2, 3])})  # type: ignore
    with pytest.raises(AssertionError):
        preds.to_npz(temp_dir / "test.npz")


def test_dataset_name_none() -> None:
    """Tests that a DataTuple can be saved without the name property."""
    datatup = DataTuple(
        x=pd.DataFrame([3.0], columns=["a1"]),
        s=pd.DataFrame([4.0], columns=["b2"]),
        y=pd.DataFrame([6.0], columns=["c3"]),
        name=None,
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        path = tmp_path / "pytest.npz"
        datatup.to_npz(path)
        # reload from feather file
        reloaded = DataTuple.from_npz(path)
    assert reloaded.name is None
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_frame_equal(datatup.s, reloaded.s)
    pd.testing.assert_frame_equal(datatup.y, reloaded.y)


def test_dataset_name_with_spaces() -> None:
    """Tests that a dataset name can contain spaces and special chars."""
    name = "This is a very@#$%^&*((())) complicated name"
    datatup = TestTuple(
        x=pd.DataFrame([3.0], columns=["a1"]), s=pd.DataFrame([4.0], columns=["b2"]), name=name
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        path = tmp_path / "pytest2.npz"
        datatup.to_npz(path)
        # reload from feather file
        reloaded = TestTuple.from_npz(path)
    assert name == reloaded.name
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_frame_equal(datatup.s, reloaded.s)


def test_apply_to_joined_df() -> None:
    """Tests apply_to_joined_df_function."""
    datatup = DataTuple(
        x=pd.DataFrame([3.0], columns=["a1"]),
        s=pd.DataFrame([4.0], columns=["b2"]),
        y=pd.DataFrame([6.0], columns=["c3"]),
        name=None,
    )

    def _identity(x: pd.DataFrame):
        return x

    result = datatup.apply_to_joined_df(_identity)
    pd.testing.assert_frame_equal(datatup.x, result.x)
    pd.testing.assert_frame_equal(datatup.s, result.s)
    pd.testing.assert_frame_equal(datatup.y, result.y)


def test_data_tuple_len() -> None:
    """Test DataTuple len property."""
    datatup_unequal_len = DataTuple(
        x=pd.DataFrame([3.0, 2.0], columns=["a1"]),
        s=pd.DataFrame([4.0], columns=["b2"]),
        y=pd.DataFrame([6.0], columns=["c3"]),
        name=None,
    )
    with pytest.raises(AssertionError):
        len(datatup_unequal_len)

    datatup_equal_len = DataTuple(
        x=pd.DataFrame([3.0, 2.0, 1.0], columns=["a1"]),
        s=pd.DataFrame([4.0, 5.0, 9.0], columns=["b2"]),
        y=pd.DataFrame([6.0, 4.2, 6.7], columns=["c3"]),
        name=None,
    )
    assert len(datatup_equal_len) == 3
