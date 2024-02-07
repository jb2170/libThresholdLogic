from typing import List, Tuple

from .BaseNeuron import BaseNeuron
from ..TransferFunction import TransferFunction, Heaviside

class Neuron(BaseNeuron):
    """
    A real neuron with weighted inputs, bias, and transfer function
    """
    def __init__(
        self,
        inputs: List[Tuple[float, BaseNeuron]],
        bias: float,
        transfer_function: TransferFunction
    ) -> None:
        self.inputs = inputs
        self.bias = bias
        self.transfer_function = transfer_function

    def do_call(self, cache) -> float:
        return self.transfer_function(
            sum(weight * input_(cache) for (weight, input_) in self.inputs) - self.bias
        )

    def add_input(self, weight: float, input_: BaseNeuron) -> None:
        self.inputs.append((weight, input_))

class Perceptron(Neuron):
    def __init__(
        self,
        inputs: List[Tuple[float, BaseNeuron]],
        bias: float
    ) -> None:
        super().__init__(inputs, bias, Heaviside(0.0 - 0.00000001))
