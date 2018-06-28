import pandas as pd


def load_data(name: str) -> pd.DataFrame:
    data_loc: str = "data/csvs/{}.csv".format(name)
    data: pd.DataFrame = pd.read_csv(data_loc)
    assert isinstance(data, pd.DataFrame)
    return data
