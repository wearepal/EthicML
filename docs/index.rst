***********************************
Welcome to EthicML's documentation!
***********************************

.. note::
   If you are looking for the pre-1.0 documentation, you can find that here: https://wearepal.ai/EthicML-0.x/

Package for evaluating the performance of methods
which aim to increase fairness, accountability and/or transparency of machine learning models.

Example
=======

.. code:: python

   from ethicml import data, metrics, models, run, plot

   results = run.evaluate_models(
       datasets=[data.Adult()],
       inprocess_models=[models.SVM(), models.Kamiran()],
       preprocess_models=[models.Upsampler()],
       metrics=[metrics.Accuracy()],
       per_sens_metrics=[metrics.ProbPos(), metrics.TPR()],
       repeats=5,
   )
   plot.plot_results(results, "Accuracy", "prob_pos_Male_0÷Male_1")


Table of contents
=================

.. toctree::
   :caption: API
   :maxdepth: 2

   ethicml/index
   ethicml.data
   ethicml.models/index
   ethicml.metrics
   ethicml.run
   ethicml.plot

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/migration_guide
   tutorials/adult_dataset



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
