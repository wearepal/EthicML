"""Collection of functions that enable parallelism."""
import asyncio
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from tqdm import tqdm

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm, InAlgorithmAsync
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm, PreAlgorithmAsync
from ethicml.utility import DataTuple, Prediction, TestTuple, TrainTestPair

__all__ = ["arrange_in_parallel", "run_in_parallel"]

_AT = TypeVar("_AT", InAlgorithmAsync, PreAlgorithmAsync)  # async algorithm type
_ST = TypeVar("_ST", InAlgorithm, PreAlgorithm)  # sync algorithm type
_RT = TypeVar("_RT", Prediction, Tuple[DataTuple, TestTuple])  # return type

RunType = Callable[[DataTuple, TestTuple], Coroutine[Any, Any, _RT]]
BlockingRunType = Callable[[DataTuple, TestTuple], _RT]
DataSeq = Sequence[TrainTestPair]


@dataclass
class _Task(Generic[_RT]):
    algo_id: int
    data_id: int
    algo: Tuple[RunType[_RT], str]
    train_test_pair: TrainTestPair


async def arrange_in_parallel(
    algos: Sequence[Tuple[RunType[_RT], str]], data: DataSeq, max_parallel: int = 0
) -> List[List[_RT]]:
    """Arrange the given algorithms to run (embarrassingly) parallel.

    Args:
        algos: list of tuples consisting of a `run_async` function of an algorithm and a name
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
    # ================================== create queue of tasks ====================================
    task_queue: asyncio.Queue[_Task[_RT]] = asyncio.Queue()
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    for i, algo in enumerate(algos):
        for j, data_item in enumerate(data):
            task_queue.put_nowait(_Task(i, j, algo, data_item))

    num_tasks = len(algos) * len(data)
    pbar: tqdm = tqdm(total=num_tasks, smoothing=0)

    # ===================================== create workers ========================================
    num_cpus = os.cpu_count()
    default_num_workers: int = num_cpus if num_cpus is not None else 1
    num_workers = max_parallel if max_parallel > 0 else default_num_workers
    result_dict: Dict[Tuple[int, int], _RT] = {}
    workers = [
        _eval_worker(worker_id, task_queue, result_dict, pbar) for worker_id in range(num_workers)
    ]
    # ====================================== run workers ==========================================
    await asyncio.gather(*workers)
    # confirm that the queue is empty
    assert task_queue.empty()
    pbar.close()  # very important! when we're not using "with", we have to close tqdm manually
    # turn dictionary into a list; the outer list is over the algos, the inner over the datasets
    # NOTE: if you want to change the return type, be warned that CrossValidator depends on it
    return [[result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]


async def _eval_worker(
    worker_id: int,
    task_queue: "asyncio.Queue[_Task[_RT]]",
    result_dict: Dict[Tuple[int, int], _RT],
    pbar: tqdm,
) -> None:
    while not task_queue.empty():
        # get a work item out of the queue
        task = task_queue.get_nowait()
        (run_algo, algo_name) = task.algo
        train, test = task.train_test_pair
        # do some logging
        logging: OrderedDict[str, str] = OrderedDict()
        logging["model"] = algo_name
        logging["dataset"] = train.name if train.name is not None else ""
        logging["worker_id"] = str(worker_id)
        pbar.set_postfix(ordered_dict=logging)
        # do the work
        result: _RT = await run_algo(train, test)
        # put result into results dictionary
        result_dict[(task.algo_id, task.data_id)] = result
        pbar.update()
        # notify the queue that the work item has been processed
        task_queue.task_done()


InSeq = Sequence[InAlgorithm]
PreSeq = Sequence[PreAlgorithm]
InResult = List[List[Prediction]]
PreResult = List[List[Tuple[DataTuple, TestTuple]]]


@overload
async def run_in_parallel(algos: InSeq, data: DataSeq, max_parallel: int = 0) -> InResult:
    ...


@overload
async def run_in_parallel(algos: PreSeq, data: DataSeq, max_parallel: int = 0) -> PreResult:
    ...


async def run_in_parallel(
    algos: Union[InSeq, PreSeq], data: DataSeq, max_parallel: int = 0
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
    if not algos:
        return cast(List[List[Prediction]], [[]])
    if isinstance(algos[0], InAlgorithm):  # pylint: disable=no-else-return  # mypy needs this
        in_algos = cast(Sequence[InAlgorithm], algos)
        # Mypy complains in the next line because of https://github.com/python/mypy/issues/5374
        in_async_algos, async_idx, in_blocking_algos = _filter(
            in_algos, InAlgorithmAsync  # type: ignore[misc]
        )
        return await _generic_run_in_parallel(
            async_algos=[(algo.run_async, algo.name) for algo in in_async_algos],
            async_idx=async_idx,
            blocking_algos=[(algo.run, algo.name) for algo in in_blocking_algos],
            data=data,
            max_parallel=max_parallel,
        )
    elif isinstance(algos[0], PreAlgorithm):
        pre_algos = cast(Sequence[PreAlgorithm], algos)
        # Mypy complains in the next line because of https://github.com/python/mypy/issues/5374
        pre_async_algos, async_idx, pre_blocking_algos = _filter(
            pre_algos, PreAlgorithmAsync  # type: ignore[misc]
        )
        return await _generic_run_in_parallel(
            async_algos=[(algo.run_async, algo.name) for algo in pre_async_algos],
            async_idx=async_idx,
            blocking_algos=[(algo.run, algo.name) for algo in pre_blocking_algos],
            data=data,
            max_parallel=max_parallel,
        )


def _filter(algos: Sequence[_ST], algo_type: Type[_AT]) -> Tuple[List[_AT], List[int], List[_ST]]:
    # Filter out those algorithms that actually can run in their own process.
    # This is unfortunately really complicated because we have to keep track of the indices
    # in order to reassemble the returned list correctly.
    filtered_algos: List[_AT] = []
    filtered_idx: List[int] = []
    remaining_algos: List[_ST] = []
    for i, algo in enumerate(algos):
        if isinstance(algo, algo_type):
            filtered_algos.append(algo)
            filtered_idx.append(i)
        else:
            remaining_algos.append(algo)
    return filtered_algos, filtered_idx, remaining_algos


async def _generic_run_in_parallel(
    async_algos: Sequence[Tuple[RunType[_RT], str]],
    async_idx: List[int],
    blocking_algos: Sequence[Tuple[BlockingRunType[_RT], str]],
    data: DataSeq,
    max_parallel: int,
) -> List[List[_RT]]:
    """Generic version of `run_in_parallel` that allows us to do this with type safety."""
    if not data:
        return []

    print("synchronous algorithms...", flush=True)  # flush to avoid conflict with tqdm
    # first get the blocking results
    pbar = tqdm(total=len(blocking_algos) * len(data), smoothing=0)
    blocking_results: List[List[_RT]] = []
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    for run, name in blocking_algos:
        temp_results: List[_RT] = []
        for train, test in data:
            logging = OrderedDict()
            logging["model"] = name
            logging["dataset"] = train.name if train.name is not None else ""
            pbar.set_postfix(ordered_dict=logging)
            temp_results.append(run(train, test))
            pbar.update()
        blocking_results.append(temp_results)
    pbar.close()  # very important! when we're not using "with", we have to close tqdm manually

    print("asynchronous algorithms...", flush=True)  # flush to avoid conflict with tqdm
    # then start the asynchronous results
    if async_algos:
        async_results = await arrange_in_parallel(async_algos, data, max_parallel)
    else:
        async_results = []

    # now reassemble everything in the right order
    results: List[List[_RT]] = []
    async_counter = 0
    blocking_counter = 0
    for i in range(len(async_algos) + len(blocking_algos)):
        if i in async_idx:
            results.append(async_results[async_counter])
            async_counter += 1
        else:
            results.append(blocking_results[blocking_counter])
            blocking_counter += 1
    return results
