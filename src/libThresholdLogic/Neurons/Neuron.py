from typing import List, Tuple

from .BaseNeuron import BaseNeuron

epsilon = 0.000001
# to account for IEEE754 mishaps when floats *almost* add up to a number,
# but sometimes end up a little short; reduce the Perceptron's threshold by epsilon

class Perceptron(BaseNeuron):
    """
    A real neuron with weighted inputs, bias, and transfer function
    """
    def __init__(self, inputs: List[Tuple[float, BaseNeuron]], bias: float) -> None:
        self.inputs = inputs
        self.bias = bias

    def do_call(self, cache) -> float:
        return self.heaviside(
            sum(weight * input_(cache) for (weight, input_) in self.inputs) - self.bias
        )

    @staticmethod
    def heaviside(f: float) -> float:
        if f >= 0.0 - epsilon:
            return 1.0
        else:
            return 0.0

    def add_input(self, weight: float, input_: BaseNeuron) -> None:
        self.inputs.append((weight, input_))
