"""Collection of functions that enable parallelism."""
import os
import asyncio
from typing import List, Tuple, Dict, Union, Sequence, overload, Generic, TypeVar, Type
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from ethicml.utility.data_structures import DataTuple, TrainTestPair, TestTuple
from ethicml.algorithms import Algorithm
from ethicml.algorithms.inprocess import InAlgorithm, InAlgorithmAsync
from ethicml.algorithms.preprocess import PreAlgorithm, PreAlgorithmAsync


_T = TypeVar("_T", InAlgorithm, InAlgorithmAsync, PreAlgorithm, PreAlgorithmAsync)
_AT = TypeVar("_AT", InAlgorithmAsync, PreAlgorithmAsync)
_RT = TypeVar("_RT", pd.DataFrame, Tuple[DataTuple, TestTuple])
# _T = TypeVar("_T", InAlgorithmAsync, PreAlgorithmAsync, covariant=True)


@dataclass
class _Task(Generic[_AT]):
    algo_id: int
    data_id: int
    algo: _AT
    train_test_pair: TrainTestPair


# ReturnType = Union[pd.DataFrame, Tuple[DataTuple, TestTuple]]
AnyAsync = Union[InAlgorithmAsync, PreAlgorithmAsync]
AnyAlgo = Union[InAlgorithm, PreAlgorithm]
InAsyncSeq = Sequence[InAlgorithmAsync]
PreAsyncSeq = Sequence[PreAlgorithmAsync]
InSeq = Sequence[InAlgorithm]
PreSeq = Sequence[PreAlgorithm]
InResult = List[List[pd.DataFrame]]
PreResult = List[List[Tuple[DataTuple, TestTuple]]]


@overload
async def arrange_in_parallel(
    algos: InAsyncSeq,
    data: List[TrainTestPair],
    max_parallel: int = 0,
) -> InResult: ...
@overload
async def arrange_in_parallel(
    algos: PreAsyncSeq,
    data: List[TrainTestPair],
    max_parallel: int = 0,
) -> PreResult: ...
async def arrange_in_parallel(
    algos: Union[InAsyncSeq, PreAsyncSeq],
    data: List[TrainTestPair],
    max_parallel: int = 0,
) -> Union[InResult, PreResult]:
    """Arrange the given algorithms to run (embarrassingly) parallel.

    Args:
        algos: list of algorithms that implement the `run_async` function
        data: list of pairs of data tuples (train and test)
        max_parallel: how many processes can run in parallel at most. if zero (or negative), then
                      there is no maximum
    Returns:
        list of the results
    """
    assert len(algos) >= 1
    assert len(data) >= 1
    assert isinstance(data[0][0], DataTuple)
    assert isinstance(data[0][1], TestTuple)

    if isinstance(algos, list) and isinstance(algos[0], InAlgorithmAsync):
        in_result_dict: Dict[Tuple[int, int], pd.DataFrame] = {}
        in_algos: InAsyncSeq = algos
        await _queue_and_start_workers(in_result_dict, in_algos, data, max_parallel)
        # turn dictionary into a list; the outer list is over the algos, the inner over the datasets
        # NOTE: if you want to change the return type, be warned that CrossValidator depends on it
        return [[in_result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]
    if isinstance(algos, list) and isinstance(algos[0], PreAlgorithmAsync):
        pre_result_dict: Dict[Tuple[int, int], Tuple[DataTuple, TestTuple]] = {}
        await _queue_and_start_workers(pre_result_dict, algos, data, max_parallel)
        return [[pre_result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]
    raise ValueError("Unsupported algorithms")


async def _queue_and_start_workers(
    result_dict: Dict[Tuple[int, int], _RT],
    algos: Sequence[_AT],
    data: List[TrainTestPair],
    max_parallel: int,
) -> None:
    # ================================== create queue of tasks ====================================
    task_queue: asyncio.Queue[_Task[_AT]] = asyncio.Queue()
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    for i, algo in enumerate(algos):
        for j, data_item in enumerate(data):
            task_queue.put_nowait(_Task(i, j, algo, data_item))
        if not isinstance(algo, (InAlgorithmAsync, PreAlgorithmAsync)):
            raise RuntimeError(f"Algorithm \"{algo.name}\" is not asynchronous!")

    num_tasks = len(algos) * len(data)
    pbar: tqdm = tqdm(total=num_tasks)

    # ===================================== create workers ========================================
    num_cpus = os.cpu_count()
    default_num_workers: int = num_cpus if num_cpus is not None else 1
    num_workers = max_parallel if max_parallel > 0 else default_num_workers
    workers = [
        _eval_worker(worker_id, task_queue, result_dict, pbar) for worker_id in range(num_workers)
    ]
    # ====================================== run workers ==========================================
    await asyncio.gather(*workers)
    # confirm that the queue is empty
    assert task_queue.empty()
    pbar.close()  # very important! when we're not using "with", we have to close tqdm manually


async def _eval_worker(
    worker_id: int,
    task_queue: "asyncio.Queue[_Task[_AT]]",
    result_dict: Dict[Tuple[int, int], _RT],
    pbar: tqdm,
) -> None:
    while not task_queue.empty():
        # get a work item out of the queue
        task = task_queue.get_nowait()
        train, test = task.train_test_pair
        # do some logging
        logging: OrderedDict[str, str] = OrderedDict()
        logging['model'] = task.algo.name
        logging['dataset'] = train.name if train.name is not None else ""
        logging['worker_id'] = str(worker_id)
        pbar.set_postfix(ordered_dict=logging)
        # do the work
        result: _RT
        if isinstance(task.algo, InAlgorithmAsync):
            result = await task.algo.run_async(train, test)  # type: ignore[assignment]
        else:
            result = await task.algo.run_async(train, test)  # type: ignore[assignment]
        # put result into results dictionary
        result_dict[(task.algo_id, task.data_id)] = result
        pbar.update()
        # notify the queue that the work item has been processed
        task_queue.task_done()

@overload
async def run_in_parallel(
    algos: InSeq, data: List[TrainTestPair], max_parallel: int = 0,
) -> InResult: ...

@overload
async def run_in_parallel(
    algos: PreSeq, data: List[TrainTestPair], max_parallel: int = 0,
) -> PreResult: ...

async def run_in_parallel(
    algos: Union[InSeq, PreSeq], data: List[TrainTestPair], max_parallel: int = 0,
) -> Union[InResult, PreResult]:
    """Run the given algorithms (embarrassingly) parallel.

    Args:
        algos: list of algorithms
        data: list of pairs of data tuples (train and test)
        max_parallel: how many processes can run in parallel at most. if zero (or negative), then
                      there is no maximum
    Returns:
        list of the results
    """
    results: Union[InResult, PreResult] = []


def _filter(algos: Sequence[Algorithm], algo_type: Type[_T]) -> Tuple[List[_T], List[int]]:
    filtered_algos: List[_T] = []
    filtered_idx: List[int] = []
    for i, algo in enumerate(algos):
        if isinstance(algo, algo_type):
            filtered_algos.append(algo)
            filtered_idx.append(i)
    return filtered_algos, filtered_idx



async def _split_run_merge(
    results: List[_RT], algos: Sequence[_AT], data: List[TrainTestPair], max_parallel: int
) -> None:
    # Filter out those algorithms that actually can run in their own process.
    # This is unfortunately really complicated because we have to keep track of the indices
    # in order to reassemble the returned list correctly.
    async_algos: List[_AT] = []
    async_idx: List[int] = []
    blocking_algos: List[AnyAlgo] = []
    blocking_idx: List[int] = []
    for i, algo in enumerate(algos):
        if isinstance(algo, (InAlgorithmAsync, PreAlgorithmAsync)):
            async_algos.append(algo)
            async_idx.append(i)
        else:
            blocking_algos.append(algo)
            blocking_idx.append(i)

    # first start the asynchronous results
    async_coroutines = arrange_in_parallel(async_algos, data, max_parallel)

    # then get the blocking results
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    blocking_results = [[algo.run(train, test) for train, test in data] for algo in blocking_algos]

    # then wait for the asynchronous results to come in
    async_results: Union[InResult, PreResult] = await async_coroutines

    # now reassemble everything in the right order
    results: Union[InResult, PreResult] = []
    async_counter = 0
    blocking_counter = 0
    for i in range(len(algos)):
        if i in async_idx:
            results.append(async_results[async_counter])
            async_counter += 1
        else:
            results.append(blocking_results[blocking_counter])
            blocking_counter += 1
    return results
