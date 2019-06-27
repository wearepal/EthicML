"""
Collection of functions that enable parallelism
"""
import os
import asyncio
from typing import List, Tuple, Dict

import pandas as pd

from ethicml.utility.data_structures import DataTuple
from ethicml.algorithms.inprocess import InAlgorithm, InAlgorithmAsync


Data = Tuple[DataTuple, DataTuple]


async def arrange_in_parallel(
    algos: List[InAlgorithmAsync], data: List[Data], max_parallel: int = 0
) -> List[List[pd.DataFrame]]:
    """Arrange the given algorithms to run (embarrassingly) parallel

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
    assert isinstance(data[0][1], DataTuple)
    # create queue of tasks
    task_queue: asyncio.Queue = asyncio.Queue()
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    for i, algo in enumerate(algos):
        for j, data_item in enumerate(data):
            task_queue.put_nowait((i, j, algo, data_item))
    # create workers
    num_cpus = os.cpu_count()
    default_num_workers: int = num_cpus if num_cpus is not None else 1
    num_workers = max_parallel if max_parallel > 0 else default_num_workers
    result_dict: Dict[Tuple[int, int], pd.DataFrame] = {}
    workers = [_eval_worker(worker_id, task_queue, result_dict) for worker_id in range(num_workers)]
    # run workers and confirm that the queue is empty
    await asyncio.gather(*workers)
    assert task_queue.empty()
    # turn dictionary into a list
    return [[result_dict[(i, j)] for j in range(len(data))] for i in range(len(algos))]


async def _eval_worker(
    worker_id: int,
    task_queue: "asyncio.Queue[Tuple[int, int, InAlgorithmAsync, Data]]",
    result_dict: Dict[Tuple[int, int], pd.DataFrame],
) -> None:
    while not task_queue.empty():
        # get a work item out of the queue
        algo_id, data_id, algo, (train, test) = task_queue.get_nowait()
        # do the work
        result = await algo.run_async(train, test)
        # put result into results dictionary
        result_dict[(algo_id, data_id)] = result
        print(f"Worker {worker_id} has finished algo {algo_id} on dataset {data_id}.")
        # notify the queue that the work item has been processed
        task_queue.task_done()


def run_in_parallel(
    algos: List[InAlgorithm], data: List[Data], max_parallel: int = -1
) -> List[List[pd.DataFrame]]:
    """Run the given algorithms (embarrassingly) parallel

    Args:
        algos: list of in-process algorithms
        data: list of pairs of data tuples (train and test)
        max_parallel: how many processes can run in parallel at most. if negative, then there is no
                      maximum
    Returns:
        list of the results
    """
    # Filter out those algorithms that actually can run in their own process.
    # This is unfortunately really complicated because we have to keep track of the indices
    # in order to reassemble the returned list correctly.
    async_algos: List[InAlgorithmAsync] = []
    async_idx: List[int] = []
    blocking_algos: List[InAlgorithm] = []
    blocking_idx: List[int] = []
    for i, algo in enumerate(algos):
        if isinstance(algo, InAlgorithmAsync):
            async_algos.append(algo)
            async_idx.append(i)
        else:
            blocking_algos.append(algo)
            blocking_idx.append(i)
    event_loop = asyncio.get_event_loop()

    # first start the asynchronous results
    async_coroutines = arrange_in_parallel(async_algos, data, max_parallel)

    # then get the blocking results
    # for each algorithm, first loop over all available datasets and then go on to the next algo
    blocking_results = [[algo.run(train, test) for train, test in data] for algo in blocking_algos]

    # then wait for the asynchronous results to come in
    async_results = event_loop.run_until_complete(async_coroutines)

    # now reassemble everything in the right order
    results: List[List[pd.DataFrame]] = []
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
