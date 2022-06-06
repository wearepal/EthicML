"""Test the saving data capability."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, List, Mapping
from typing_extensions import Final, final

import numpy as np
import pandas as pd
import pytest

from ethicml import DataTuple, HyperParamType, Prediction, SubgroupTuple
from ethicml.models import InAlgoArgs, InAlgorithmSubprocess

NPZ: Final[str] = "test.npz"


def test_simple_saving() -> None:
    """Tests that a DataTuple can be saved."""
    data_tuple = DataTuple.from_df(
        x=pd.DataFrame({"a1": np.array([9.4, np.nan, 0.0])}),
        s=pd.Series([18, -3, int(1e10)], name="b1"),
        y=pd.Series([0, 1, 0], name="c3"),
        name="test data",
    )

    @dataclass
    class CheckEquality(InAlgorithmSubprocess):
        """Dummy algorithm class for testing whether writing and reading feather files works."""

        is_fairness_algo: ClassVar[bool] = False

        @final
        def get_name(self) -> str:
            return "Check equality"

        @final
        def get_hyperparameters(self) -> HyperParamType:
            return {}

        def _script_command(self, in_algo_args: InAlgoArgs):  # type: ignore[misc]
            """Check if the dataframes loaded from the files are the same as the original ones."""
            assert in_algo_args["mode"] == "run", "model doesn't support the fit/predict split yet"
            loaded = DataTuple.from_file(Path(in_algo_args["train"]))
            pd.testing.assert_frame_equal(data_tuple.x, loaded.x)
            pd.testing.assert_series_equal(data_tuple.s, loaded.s)
            pd.testing.assert_series_equal(data_tuple.y, loaded.y)
            # write a file for the predictions
            np.savez(in_algo_args["predictions"], hard=np.load(in_algo_args["train"])["y"])
            return ["-c", "pass"]

        def _get_path_to_script(self) -> List[str]:
            return []

        def _get_flags(self) -> Mapping[str, Any]:
            return {}

    data_y = CheckEquality().run(data_tuple, data_tuple, -1)
    pd.testing.assert_series_equal(data_tuple.y, data_y.hard, check_names=False)


def test_predictions_loaded(temp_dir: Path) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]))
    preds.save_to_file(temp_dir / NPZ)
    loaded = Prediction.from_file(temp_dir / NPZ)
    pd.testing.assert_series_equal(preds.hard, loaded.hard, check_dtype=False)


def test_predictions_info_loaded(temp_dir: Path) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]), info={"sample": 123.4})
    preds.save_to_file(temp_dir / NPZ)
    loaded = Prediction.from_file(temp_dir / NPZ)
    pd.testing.assert_series_equal(preds.hard, loaded.hard, check_dtype=False)
    assert preds.info == loaded.info


def test_predictions_info_loaded_bad(temp_dir: Path) -> None:
    """Test that predictions can be saved and loaded."""
    preds = Prediction(hard=pd.Series([1]), info={"sample": np.array([1, 2, 3])})  # type: ignore
    with pytest.raises(AssertionError):
        preds.save_to_file(temp_dir / NPZ)


def test_dataset_name_none() -> None:
    """Tests that a DataTuple can be saved without the name property."""
    datatup = DataTuple.from_df(
        x=pd.DataFrame([3.0], columns=["a1"]),
        s=pd.Series([4.0], name="b2"),
        y=pd.Series([6.0], name="c3"),
        name=None,
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        path = tmp_path / "pytest.npz"
        datatup.save_to_file(path)
        # reload from feather file
        reloaded = DataTuple.from_file(path)
    assert reloaded.name is None
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_series_equal(datatup.s, reloaded.s)
    pd.testing.assert_series_equal(datatup.y, reloaded.y)


def test_dataset_name_with_spaces() -> None:
    """Tests that a dataset name can contain spaces and special chars."""
    name = "This is a very@#$%^&*((())) complicated name"
    datatup = SubgroupTuple.from_df(
        x=pd.DataFrame([3.0], columns=["a1"]), s=pd.Series([4.0], name="b2"), name=name
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        path = tmp_path / "pytest2.npz"
        datatup.save_to_file(path)
        # reload from feather file
        reloaded = SubgroupTuple.from_file(path)
    assert name == reloaded.name
    pd.testing.assert_frame_equal(datatup.x, reloaded.x)
    pd.testing.assert_series_equal(datatup.s, reloaded.s)


def test_apply_to_joined_df() -> None:
    """Tests apply_to_joined_df_function."""
    datatup = DataTuple.from_df(
        x=pd.DataFrame([3.0], columns=["a1"]),
        s=pd.Series([4.0], name="b2"),
        y=pd.Series([6.0], name="c3"),
        name=None,
    )

    def _identity(x: pd.DataFrame):
        return x

    result = datatup.apply_to_joined_df(_identity)
    pd.testing.assert_frame_equal(datatup.x, result.x)
    pd.testing.assert_series_equal(datatup.s, result.s)
    pd.testing.assert_series_equal(datatup.y, result.y)


def test_data_tuple_len() -> None:
    """Test DataTuple len property."""
    x = pd.DataFrame([3.0, 2.0], columns=["a1"])
    s = pd.Series([4.0], name="b2")
    y = pd.Series([6.0], name="c3")
    name = None
    with pytest.raises(AssertionError):
        DataTuple.from_df(x=x, s=s, y=y, name=name)

    datatup_equal_len = DataTuple.from_df(
        x=pd.DataFrame([3.0, 2.0, 1.0], columns=["a1"]),
        s=pd.Series([4.0, 5.0, 9.0], name="b2"),
        y=pd.Series([6.0, 4.2, 6.7], name="c3"),
        name=None,
    )
    assert len(datatup_equal_len) == 3
