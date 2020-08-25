Welcome to EthicML's documentation!
===================================

Package for evaluating the performance of methods
which aim to increase fairness, accountability and/or transparency of machine learning models.

Example
-------

.. code:: python

   import ethicml as eml

   results = eml.evaluate_models(
       datasets=[eml.adult()],
       inprocess_models=[eml.SVM(), eml.Kamiran()],
       preprocess_models=[eml.Upsampler()],
       metrics=[eml.Accuracy()],
       per_sens_attribute=[eml.ProbPos(), eml.TPR()],
       repeats=5
   )
   eml.plot_results(results, "Accuracy", "prob_pos_Male_0/Male_1")


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
