"""This module contains metrics which can be applied to prediction results.

Some example code

.. code:: python

   import ethicml as em

   em.run_metrics(predictions, test_data, metrics=[em.Accuracy(), em.TPR(), em.ProbPos()])
"""
from .accuracy import *
from .anti_spur import AS
from .average_odds import AverageOddsDiff
from .balanced_accuracy import BalancedAccuracy
from .bcr import BCR
from .confusion_matrix import LabelOutOfBounds, confusion_matrix
from .cv import CV, AbsCV
from .dependence_measures import *
from .fnr import FNR
from .fpr import FPR
from .hsic import Hsic
from .metric import Metric
from .npv import NPV
from .ppv import PPV
from .prob_neg import ProbNeg
from .prob_outcome import ProbOutcome
from .prob_pos import ProbPos
from .theil import Theil
from .tnr import TNR
from .tpr import TPR
