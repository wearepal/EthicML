***********************************
Welcome to EthicML's documentation!
***********************************

Package for evaluating the performance of methods
which aim to increase fairness, accountability and/or transparency of machine learning models.

Example
=======

.. code:: python

   import ethicml as em
   import ethicml.data as emda
   from ethicml import metrics, models

   results = em.evaluate_models(
       datasets=[emda.Adult()],
       inprocess_models=[models.SVM(), models.Kamiran()],
       preprocess_models=[models.Upsampler()],
       metrics=[metrics.Accuracy()],
       per_sens_metrics=[metrics.ProbPos(), metrics.TPR()],
       repeats=5,
   )
   em.plot_results(results, "Accuracy", "prob_pos_Male_0/Male_1")


API
---

.. toctree::
   :maxdepth: 2

   ethicml/index
   ethicml.data
   ethicml.models/index
   ethicml.metrics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
