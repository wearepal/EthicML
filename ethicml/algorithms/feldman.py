import pandas as pd
from typing import Dict
from BlackBoxAuditing.repairers.GeneralRepairer import Repairer

from ethicml.algorithms.algorithm import Algorithm


class Feldman(Algorithm):
    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:


        repaired_train_df = self.repair(
            pd.concat([train['x'], train['s']], axis=1),
            train['s'],
            1,
            1.0)
        repaired_test_df = self.repair(
            pd.concat([test['x'], test['s']], axis=1),
            test['s'],
            1,
            1.0)

        return repaired_train_df[train['x'].columns], repaired_test_df[train['x'].columns]


    def get_name(self) -> str:
        return "Feldman"

    def repair(self, data_df, single_sensitive, class_attr, repair_level):
        types = data_df.dtypes
        data = data_df.values.tolist()

        index_to_repair = data_df.shape[1] - single_sensitive.shape[1]
        headers = data_df.columns.tolist()
        repairer = Repairer(data, index_to_repair, repair_level, False)
        data = repairer.repair(data)

        # The repaired data no longer includes its headers.
        data_df = pd.DataFrame(data, columns = headers)
        data_df = data_df.astype(dtype=types)

        return data_df
