"""
Implementation of Heaviside Activation function
"""

import numpy
from utility import Activation


class Heaviside(Activation):

    def apply(self, soft_output: numpy.array) -> numpy.array:
        heavi = lambda x: 1 if x >= 0.5 else 0
        return heavi(soft_output)

    def get_name(self) -> str:
        return "Heaviside"
