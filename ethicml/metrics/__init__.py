"""This module contains metrics which can be applied to prediction results.

Some example code

.. code:: python

   from ethicml.metrics import Accuracy, TPR, ProbPos
   from ethicml.evaluators import run_metrics

   run_metrics(predictions, test_data, metrics=[Accuracy, TPR, ProbPos])
"""
from .accuracy import *
from .anti_spur import AS
from .balanced_accuracy import BalancedAccuracy
from .bcr import BCR
from .confusion_matrix import LabelOutOfBounds, confusion_matrix
from .cv import CV, AbsCV
from .hsic import Hsic
from .metric import Metric
from .dependence_measures import *
from .npv import NPV
from .ppv import PPV
from .prob_neg import ProbNeg
from .prob_outcome import ProbOutcome
from .prob_pos import ProbPos
from .theil import Theil
from .tnr import TNR
from .tpr import TPR
