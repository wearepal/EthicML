"""Zemel's Learned Fair Representations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Type, TypeVar, Union

import aif360
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import StandardDataset
from ranzen import implements

from ethicml.utility import DataTuple, TestTuple

from .pre_algorithm import PreAlgorithm

T = TypeVar("T", DataTuple, TestTuple)

__all__ = ["ZemelAif"]


class AifPreAlgorithm(PreAlgorithm):
    def em_to_aif360(self, data: DataTuple) -> aif360.datasets.StandardDataset:
        self.target_cols = data.y.columns
        self.sens_cols = data.s.columns

        df = pd.concat([data.x, data.s, data.y], axis="columns")

        return StandardDataset(
            df=df,
            label_name=data.y.columns[0],
            favorable_classes=lambda x: x > 0,
            protected_attribute_names=data.s.columns,
            privileged_classes=[lambda x: x == 1],
            categorical_features=[],
        )

    def aif360_to_em(self, data: StandardDataset) -> DataTuple:
        sens_feats = np.reshape(
            data.protected_attributes[
                :, data.protected_attribute_names.index(self.alg.protected_attribute_name)
            ],
            [-1, 1],
        )
        sens_feats = pd.DataFrame(sens_feats, columns=[self.sens_cols])

        target_feats = pd.DataFrame(data.labels, columns=[self.target_cols])

        return DataTuple(x=data.features, s=sens_feats, y=target_feats)

    def __init__(
        self,
        aif360_algo: Type[aif360.Transformer],
        fit_extra_args: Dict[str, Any],
        name: str,
        seed: int,
        init_extra_args: Dict[str, Any],
    ):
        super().__init__(name, seed, out_size=None)
        self._aif360_algo = aif360_algo
        self._fit_extra_args = fit_extra_args
        self._init_args = init_extra_args

    @implements(PreAlgorithm)
    def fit(self, train_data: DataTuple) -> AifPreAlgorithm:
        unprivileged_group = [{train_data.s.columns[0]: 0}]
        privileged_group = [{train_data.s.columns[0]: 1}]

        self.alg = self._aif360_algo(
            privileged_groups=privileged_group,
            unprivileged_groups=unprivileged_group,
            **self._init_args,
        )

        aif360_dataset = self.em_to_aif360(train_data)
        self.alg = self.alg.fit(aif360_dataset, **self._fit_extra_args)
        return self

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        aif360_dataset = self.em_to_aif360(data)
        result = self.alg.predict(aif360_dataset)
        return self.aif360_to_em(result)

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        self._out_size = train.x.shape[1]
        unprivileged_group = [{train.s.columns[0]: 0}]
        privileged_group = [{train.s.columns[0]: 1}]

        self.alg = self._aif360_algo(
            privileged_groups=privileged_group,
            unprivileged_groups=unprivileged_group,
            **self._init_args,
        )

        train = self.em_to_aif360(train)
        self.alg = self.alg.fit(train, **self._fit_extra_args)
        test = self.em_to_aif360(test)
        train = self.alg.transform(train)
        test = self.alg.transform(test)
        return self.aif360_to_em(train), self.aif360_to_em(test)


class ZemelAif(AifPreAlgorithm):
    """AIF360 implementation of Zemel's LFR."""

    def __init__(
        self,
        dir: Union[str, Path],
        threshold: float = 0.5,
        clusters: int = 2,
        Ax: float = 0.01,
        Ay: float = 0.1,
        Az: float = 0.5,
        max_iter: int = 5_000,
        maxfun: int = 5_000,
        epsilon: float = 1e-5,
        seed: int = 888,
    ) -> None:
        super().__init__(
            name="Zemel",
            seed=seed,
            aif360_algo=LFR,
            fit_extra_args={},
            init_extra_args={
                "k": clusters,
                "Ax": Ax,
                "Ay": Ay,
                "Az": Az,
                # "epsilon": epsilon,
                "verbose": 0,
                # "max_iter": max_iter,
                # "maxfun": maxfun,
                # "threshold": threshold,
            },
        )
