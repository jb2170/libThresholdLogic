from typing import List, Tuple

from .Neurons import BaseNeuron, ConstNeuron, ProxyNeuron

class NeuronNetwork:
    """
    The abstract base class for a Neuron Network
    Child classes should extend `__init__`
    """
    def __init__(
        self,
        neurons: List[BaseNeuron],
        input_layer: List[ProxyNeuron],
        output_layer: List[BaseNeuron],
    ) -> None:
        """
        Child classes should use __init__ as the function to create and connect up the
        network's neurons, and call super().__init__(...) to store them as member variables here
        """
        self.neurons = neurons
        self.input_layer = input_layer
        self.output_layer = output_layer

    def __call__(self, *inputs: float) -> Tuple[float]:
        """
        For calling the network directly with float inputs.
        If chaining neuron networks together then this function should not be called;
        connect up the output layer neurons as desired instead.
        """
        assert len(inputs) == len(self.input_layer)

        cache = {}

        for input_value, neuron_input in zip(inputs, self.input_layer):
            neuron_input.source = ConstNeuron(input_value)

        return tuple((neuron(cache) for neuron in self.output_layer))
