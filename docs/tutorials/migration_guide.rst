**********************
Migration guide to 1.0
**********************

There were a number of breaking changes from pre-1.0 versions to EthicML 1.0.

Modularization of imports
=========================

You can no longer import datasets, metrics or algorithms directrly from the top-level ``ethicml`` namespace.
There are now 4 namespaces:

- ``ethicml``

  * Evaluation helpers

  * Visualization helpers

  * Data structures (DataTuple, etc.)

- ``ethicml.models``

  * All the algorithms

- ``ethicml.metrics``

  * All the metrics

- ``ethicml.data``

  * All the datasets
