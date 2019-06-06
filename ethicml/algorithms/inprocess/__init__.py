"""In-process algorithms take training data and make predictions"""
from .agarwal_reductions import Agarwal
from .in_algorithm import InAlgorithm, InAlgorithmAsync
from .fair_gpyt import GPyT, GPyTDemPar, GPyTEqOdds
from .kamishima import Kamishima
from .kamiran import Kamiran
from .logistic_regression import LR
from .logistic_regression_cross_validated import LRCV
from .logistic_regression_probability import LRProb
from .svm import SVM
from .majority import Majority
from .mlp import MLP
