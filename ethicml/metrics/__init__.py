"""This module contains metrics which can be applied to prediction results.

Some example code

.. code:: python

   import ethicml as em

   em.run_metrics(predictions, test_data, metrics=[em.Accuracy(), em.TPR(), em.ProbPos()])
"""
from .accuracy import *
from .anti_spur import *
from .average_odds import *
from .balanced_accuracy import *
from .bcr import *
from .confusion_matrix import *
from .cv import *
from .dependence_measures import *
from .fnr import *
from .fpr import *
from .hsic import *
from .metric import *
from .npv import *
from .per_sensitive_attribute import *
from .ppv import *
from .prob_neg import *
from .prob_outcome import *
from .prob_pos import *
from .theil import *
from .tnr import *
from .tpr import *
