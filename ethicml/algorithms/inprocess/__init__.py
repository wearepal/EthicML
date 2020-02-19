"""In-process algorithms take training data and make predictions."""
from .agarwal_reductions import Agarwal
from .in_algorithm import InAlgorithm, InAlgorithmAsync
from .installed_model import InstalledModel
from .kamiran import Kamiran
from .kamishima import Kamishima
from .logistic_regression import LR, LRCV, LRProb
from .majority import Majority
from .manual import Corels
from .mlp import MLP
from .svm import SVM
from .zafar import ZafarAccuracy, ZafarBaseline, ZafarEqOdds, ZafarEqOpp, ZafarFairness
