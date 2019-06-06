"""
Collection of functions that enable parallelism
"""
import asyncio
from typing import List, Tuple, Union, Coroutine, Any

import pandas as pd

from ethicml.algorithms.utils import DataTuple
from ethicml.algorithms.inprocess import InAlgorithm, InAlgorithmSync


Data = Tuple[DataTuple, DataTuple]


async def arrange_in_parallel(
    algos: List[InAlgorithm], data: Union[Data, List[Data]]
) -> List[pd.DataFrame]:
    """Arrange the given algorithms to run (embarrassingly) parallel

    Args:
        algos: list of algorithms that implement the `run_async` function
        data: either a single pair of data tuples (train and test) or a list of pairs that has the
              same length as the list of algorithms
    Returns:
        list of the results
    """
    coroutines: List[Coroutine[Any, Any, pd.DataFrame]]
    if isinstance(data, list):
        assert len(data) == len(algos)
        coroutines = [algo.run_async(train, test) for algo, (train, test) in zip(algos, data)]
    else:
        assert len(data) == 2
        assert isinstance(data[0], DataTuple)
        assert isinstance(data[1], DataTuple)
        train, test = data
        coroutines = [algo.run_async(train, test) for algo in algos]
    return await asyncio.gather(*coroutines)


def run_in_parallel(
    algos: List[InAlgorithmSync], data: Union[Data, List[Data]]
) -> List[pd.DataFrame]:
    """Run the given algorithms (embarrassingly) parallel

    Args:
        algos: list of in-process algorithms
        data: either a single pair of data tuples (train and test) or a list of pairs that has the
              same length as the list of algorithms
    Returns:
        list of the results
    """
    # Filter out those algorithms that actually can run in their own process.
    # This is unfortunately really complicated because we have to keep track of the indices
    # in order to reassemble the returned list correctly.
    async_algos: List[InAlgorithm] = []
    async_idx: List[int] = []
    blocking_algos: List[InAlgorithmSync] = []
    blocking_idx: List[int] = []
    for i, algo in enumerate(algos):
        # naming is a bit unfortunate right now: InAlgorithmSync is the parent class of InAlgorithm
        if isinstance(algo, InAlgorithm):
            async_algos.append(algo)
            async_idx.append(i)
        else:
            blocking_algos.append(algo)
            blocking_idx.append(i)
    async_data: Union[Data, List[Data]]
    if isinstance(data, list):
        async_data = [data[i] for i in async_idx]
    else:
        async_data = data
    event_loop = asyncio.get_event_loop()

    # first start the asynchronous results
    async_coroutines = arrange_in_parallel(async_algos, async_data)

    # then get the blocking results
    if isinstance(data, list):
        blocking_results = [
            algo.run(data[i][0], data[i][1]) for algo, i in zip(blocking_algos, blocking_idx)
        ]
    else:
        train, test = data
        blocking_results = [algo.run(train, test) for algo in blocking_algos]

    # then wait for the asynchronous results to come in
    async_results = event_loop.run_until_complete(async_coroutines)

    # now reassemble everything in the right order
    results: List[pd.DataFrame] = []
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
