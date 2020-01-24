.. EthicML documentation master file, created by
   sphinx-quickstart on Tue Sep 17 14:06:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EthicML's documentation!
===================================

Package for evaluating the performance of methods which aim to increase fairness, accountability and/or transparency.

Example
-------

.. code:: python

   from ethicml.data import Adult
   from ethicml.algorithms.inprocess import SVM, Kamiran
   from ethicml.metrics import Accuracy, TPR, ProbPos
   from ethicml.evaluators import evaluate_models
   
   evaluate_models(
       datasets=[Adult()],
       inprocess_models=[SVM(), Kamiran()],
       metrics=[Accuracy()],
       per_sens_metrics=[TPR(), ProbPos()],
       repeats=5,
   )


API
---

.. toctree::
   :maxdepth: 1

   ethicml.algorithms
   ethicml.algorithms.preprocess
   ethicml.algorithms.inprocess
   ethicml.algorithms.postprocess
   ethicml.data
   ethicml.evaluators
   ethicml.metrics
   ethicml.preprocessing
   ethicml.utility
   ethicml.vision
   ethicml.visualisation



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
