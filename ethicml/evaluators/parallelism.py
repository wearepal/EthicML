"""Collection of functions that enable parallelism."""
from typing import List, Optional, Sequence, Tuple, TypeVar, Union, cast, overload
from typing_extensions import Protocol

import ray
from ray.actor import ActorHandle

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.utility import DataTuple, Prediction, TestTuple, TrainTestPair

from .progressbar import ProgressBar

__all__ = ["arrange_in_parallel", "run_in_parallel"]


InSeq = Sequence[InAlgorithm]
PreSeq = Sequence[PreAlgorithm]
InResult = List[List[Prediction]]
PreResult = List[List[Tuple[DataTuple, TestTuple]]]
DataSeq = Sequence[TrainTestPair]


@overload
def run_in_parallel(algos: InSeq, data: DataSeq, num_cpus: int = 0) -> InResult:
    ...


@overload
def run_in_parallel(algos: PreSeq, data: DataSeq, num_cpus: int = 0) -> PreResult:
    ...


def run_in_parallel(
    algos: Union[InSeq, PreSeq], data: DataSeq, num_cpus: int = 0
) -> Union[InResult, PreResult]:
    """Run the given algorithms (embarrassingly) parallel.

    Args:
        algos: list of algorithms
        data: list of pairs of data tuples (train and test)
        num_cpus: how many processes can run in parallel at most. if zero (or negative), then
            there is no maximum

    Returns:
        list of the results
    """
    if not algos or not data:
        return cast(List[List[Prediction]], [[]])
    # The following isinstance check is not at all reliable because `InAlgorithm` is a Protocol,
    # but that's completely fine because this check is only here for mypy anyway.
    if isinstance(algos[0], InAlgorithm):
        in_algos = cast(Sequence[InAlgorithm], algos)
        return arrange_in_parallel(algos=in_algos, data=data, num_cpus=num_cpus)
    else:
        pre_algos = cast(Sequence[PreAlgorithm], algos)
        return arrange_in_parallel(algos=pre_algos, data=data, num_cpus=num_cpus)


_RT = TypeVar("_RT", Prediction, Tuple[DataTuple, TestTuple], covariant=True)  # the return type


class Algorithm(Protocol[_RT]):
    """This protocol is a clever way to make `arrange_in_parallel` generic."""

    def run(self, train: DataTuple, test: TestTuple) -> _RT:
        ...


def arrange_in_parallel(
    algos: Sequence[Algorithm[_RT]], data: DataSeq, num_cpus: Optional[int] = None
) -> List[List[_RT]]:
    """Arrange the given algorithms to run (embarrassingly) parallel.

    Args:
        algos: list of tuples consisting of a `run_async` function of an algorithm and a name
        data: list of pairs of data tuples (train and test)
        num_cpus: number of CPUs to use. `None` means all.

    Returns:
        list of the results
    """
    ray.shutdown()  # (not sure why this is needed but it seems to be needed...)
    ray.init(num_cpus=num_cpus)
    assert len(algos) >= 1
    assert len(data) >= 1
    assert isinstance(data[0][0], DataTuple)
    assert isinstance(data[0][1], TestTuple)
    # ================================== create queue of tasks ====================================
    num_tasks = len(algos) * len(data)
    pbar = ProgressBar(total=num_tasks)
    futures: List[_RT] = []
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    for algo in algos:
        for data_item in data:
            futures.append(_run.remote(algo, data_item, pbar.actor))

    pbar.print_until_done()
    # actually run everything
    results = ray.get(futures)
    ray.shutdown()
    # return [[result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]
    # we have to reconstruct the nested list from the flattened list
    return [[results[i * len(data) + j] for j in range(len(data))] for i in range(len(algos))]


@ray.remote
def _run(algo: Algorithm[_RT], train_test_pair: TrainTestPair, pbar: ActorHandle) -> _RT:
    train, test = train_test_pair
    # do the work
    result: _RT = algo.run(train, test)
    # put result into results dictionary
    pbar.update.remote(1)
    return result
