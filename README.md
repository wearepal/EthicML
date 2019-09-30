# EthicML

EthicML exists to solve the problems we've found with off-the-shelf fairness comparison packages.

These other packages are useful, but given that we primarily do research,
a lot of the work we do doesn't fit into some nice box.
For example, we might want to use a 'fair' pre-processing method on the data before training a classifier on it.
We may still be experimenting and only want part of the framework to execute,
or we may want to do hyper-parameter optimization.
Whilst other frameworks can be modified to do these tasks,
you end up with hacked-together approaches that don't lend themselves to be built on in the future.
Because of this,
we're drawing a line in the sand with some of the other frameworks we've used and building our own.

### Why not use XXX?

There are an increasing number of other options,
IBM's fair-360, Aequitas, EthicalML/XAI, Fairness-Comparison and others.
They're all great at what they do, they're just not right for us.
We will however be influenced by them.

## Installation

EthicML requires Python >= 3.6.
To install EthicML, just do
```
pip3 install ethicml
```

**Attention**: In order to use all features of EthicML, PyTorch needs to be installed separately.
We are not including PyTorch as a requirement of EthicML,
because there are many different versions for different systems.

## Documentation

The documentation can be found here: https://predictive-analytics-lab.github.io/EthicML/

## Design Principles

### The Triplet

Given that we're considering fairness, the base of the toolbox is the triplet {x, s, y}

- X - Features
- S - Sensitive Label
- Y - Class Label

All methods must assume S and Y are multi-class.

We use a named tuple to contain the triplet

```python
triplet = DataTuple(x: pandas.DataFrame, s: pandas.DataFrame, y: pandas.DataFrame)
```

The dataframe may be a little inefficient,
but given the amount of splicing on conditions that we're doing, it feels worth it.

### Separation of Methods

We purposefully keep pre, during and post algorithm methods separate. This is because they have different return types.

```python
pre_algorithm.run(train: DataTuple, test: DataTuple)  # -> Tuple[pandas.DataFrame, pandas.DataFrame]
in_algorithm.run(train: DataTuple, test: DataTuple)  # -> pandas.DataFrame
post_algorithm.run(preds: DataFrame, test: DataTuple)  # -> pandas.DataFrame
```
where preds is a one column dataframe with the column name 'preds'.

### General Rules of Thumb

- Mutable data structures are bad.
- At the very least, functions should be Typed.
- Readability > Efficiency
- Don't get around warnings by just turning them off.

## Future Plans

Hopefully EthicML becomes a super easy way to look at the biases in different datasets
and get a comparison of different models.

## Development
Install development dependencies with `pip install -e .[dev]`

To use the pre-commit hooks run `pre-commit install`
