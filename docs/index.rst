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
   import ethicml.metrics as emme
   import ethicml.models as emmo

   results = em.evaluate_models(
       datasets=[emda.Adult()],
       inprocess_models=[emmo.SVM(), emmo.Kamiran()],
       preprocess_models=[emmo.Upsampler()],
       metrics=[emme.Accuracy()],
       per_sens_metrics=[emme.ProbPos(), emme.TPR()],
       repeats=5,
   )
   em.plot_results(results, "Accuracy", "prob_pos_Male_0/Male_1")


API
---

.. toctree::
   :maxdepth: 2

   ethicml/index
   ethicml.data
   models/index
   ethicml.metrics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
