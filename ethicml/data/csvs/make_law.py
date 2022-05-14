"""Make the law admissions dataset."""
import pandas as pd


def main() -> None:
    """Make the Law Dataset."""
    data = pd.read_csv("raw/law_data.csv", index_col=0).reset_index(drop=True)
    data = data[["LSAT", "UGPA", "ZFYA", "race", "sex", "first_pf"]]
    data["first_pf"] = data["first_pf"].astype(int)
    data = data.rename(columns={"race": "Race", "sex": "Sex", "first_pf": "PF"})
    data = pd.get_dummies(data, columns=["Race", "Sex", "PF"])
    data = data.sample(frac=1.0, random_state=888).reset_index(drop=True)
    compression_opts = dict(method='zip', archive_name='law.csv')
    print(data.head())
    data.to_csv("./law.csv.zip", index=False, compression=compression_opts)


if __name__ == '__main__':
    main()
