"""In-process algorithms take training data and make predictions."""
__all__ = [
    "Agarwal",
    "InAlgorithm",
    "InAlgorithmAsync",
    "InstalledModel",
    "GPyT",
    "GPyTDemPar",
    "GPyTEqOdds",
    "Kamishima",
    "Kamiran",
    "LR",
    "LRCV",
    "LRProb",
    "SVM",
    "Majority",
    "MLP",
    "Corels",
    "ZafarAccuracy",
    "ZafarBaseline",
    "ZafarEqOdds",
    "ZafarEqOpp",
    "ZafarFairness",
]
from .agarwal_reductions import Agarwal
from .in_algorithm import InAlgorithm, InAlgorithmAsync
from .installed_model import InstalledModel
from .fair_gpyt import GPyT, GPyTDemPar, GPyTEqOdds
from .kamishima import Kamishima
from .kamiran import Kamiran
from .logistic_regression import LR, LRCV, LRProb
from .svm import SVM
from .majority import Majority
from .mlp import MLP
from .manual import Corels
from .zafar import ZafarAccuracy, ZafarBaseline, ZafarEqOdds, ZafarEqOpp, ZafarFairness
