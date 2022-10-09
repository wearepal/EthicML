"""Collection of functions that enable parallelism."""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Protocol, Sequence, Tuple, TypeVar, cast, overload
from typing_extensions import TypeAlias

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ethicml.models.inprocess.in_algorithm import InAlgorithm
from ethicml.models.preprocess.pre_algorithm import PreAlgorithm
from ethicml.utility.data_structures import DataTuple, Prediction, SubgroupTuple, TrainValPair

__all__ = ["arrange_in_parallel", "run_in_parallel"]


if TYPE_CHECKING:
    InSeq: TypeAlias = Sequence[InAlgorithm]
    PreSeq: TypeAlias = Sequence[PreAlgorithm]
    InResult: TypeAlias = List[List[Prediction]]
    PreResult: TypeAlias = List[List[Tuple[DataTuple, DataTuple]]]
    DataSeq: TypeAlias = Sequence[TrainValPair]


@overload
def run_in_parallel(
    algos: InSeq, *, data: DataSeq, seeds: list[int], num_jobs: int | None = None
) -> InResult:
    ...


@overload
def run_in_parallel(
    algos: PreSeq, *, data: DataSeq, seeds: list[int], num_jobs: int | None = None
) -> PreResult:
    ...


def run_in_parallel(
    algos: InSeq | PreSeq, *, data: DataSeq, seeds: list[int], num_jobs: int | None = None
) -> InResult | PreResult:
    """Run the given algorithms (embarrassingly) parallel.

    :param algos: list of algorithms
    :param data: list of pairs of data tuples (train and test)
    :param seeds: list of seeds to use when running the model
    :param num_jobs: how many jobs can run in parallel at most. if None, use number of CPUs (Default: None)
    :returns: list of the results
    """
    if not algos or not data:
        return cast(List[List[Prediction]], [[]])
    # The following isinstance check is not at all reliable because `InAlgorithm` is a Protocol,
    # but that's completely fine because this check is only here for mypy anyway.
    if isinstance(algos[0], InAlgorithm):
        in_algos = cast(Sequence[InAlgorithm], algos)
        return arrange_in_parallel(algos=in_algos, data=data, seeds=seeds, num_jobs=num_jobs)
    else:
        pre_algos = cast(Sequence[PreAlgorithm], algos)
        # the following line is needed to help mypy along
        generic_algos: Sequence[Algorithm[tuple[DataTuple, DataTuple]]] = pre_algos
        return arrange_in_parallel(algos=generic_algos, data=data, seeds=seeds, num_jobs=num_jobs)


_RT = TypeVar("_RT", Prediction, Tuple[DataTuple, DataTuple], covariant=True)  # the return type


class Algorithm(Protocol[_RT]):
    """Protocol for making `arrange_in_parallel` generic."""

    def run(self, train: DataTuple, test: DataTuple, seed: int) -> _RT:
        ...


def arrange_in_parallel(
    algos: Sequence[Algorithm[_RT]], data: DataSeq, seeds: list[int], num_jobs: int | None = None
) -> list[list[_RT]]:
    """Arrange the given algorithms to run (embarrassingly) parallel.

    :param algos: List of tuples consisting of a `run_async` function of an algorithm and a name.
    :param data: List of pairs of data tuples (train and test).
    :param seeds: List of random seeds.
    :param num_jobs: Number of parallel jobs. `None` means as many as available CPUs.
        (Default: None)
    :returns: list of the results
    """
    runner = Parallel(n_jobs=num_jobs, verbose=10, backend="loky")
    assert len(algos) >= 1
    assert len(data) >= 1
    assert len(seeds) == len(data)
    assert isinstance(data[0][0], DataTuple)
    assert isinstance(data[0][1], (DataTuple, SubgroupTuple))
    # ================================== create queue of tasks ====================================
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    results = runner(
        _run(algo, data_item, seed) for algo in algos for (data_item, seed) in zip(data, seeds)
    )
    # return [[result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]
    # we have to reconstruct the nested list from the flattened list
    return [[results[i * len(data) + j] for j in range(len(data))] for i in range(len(algos))]


@delayed
def _run(algo: Algorithm[_RT], train_test_pair: TrainValPair, seed: int) -> _RT:
    train, test = train_test_pair
    # do the work
    try:
        result: _RT = algo.run(train, test, seed)
    except RuntimeError:
        result = Prediction(hard=pd.Series([np.NaN] * len(test)))
    if isinstance(result, Prediction):
        result.info["model_seed"] = seed
    return result
