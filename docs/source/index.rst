***********************************
Welcome to EthicML's documentation!
***********************************

Package for evaluating the performance of methods
which aim to increase fairness, accountability and/or transparency of machine learning models.

Example
=======

.. code:: python

   import ethicml as em

   results = em.evaluate_models(
       datasets=[em.adult()],
       inprocess_models=[em.SVM(), em.Kamiran()],
       preprocess_models=[em.Upsampler()],
       metrics=[em.Accuracy()],
       per_sens_metrics=[em.ProbPos(), em.TPR()],
       repeats=5,
   )
   em.plot_results(results, "Accuracy", "prob_pos_Male_0/Male_1")


API
===

.. toctree::
   :maxdepth: 3

   algorithms/index
   data/index
   evaluators
   metrics/index
   preprocessing
   utility
   vision
   visualisation



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
