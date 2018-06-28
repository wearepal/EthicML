import pandas as pd
from data.load import load_data


class TestClass(object):
    def test_can_load_test_data(self):
        data_loc: str = "data/csvs/test.csv"
        data: pd.DataFrame = pd.read_csv(data_loc)
        assert data is not None

    def test_load_data(self):
        data: pd.DataFrame = load_data('test')
        assert data.shape == (200,3)
