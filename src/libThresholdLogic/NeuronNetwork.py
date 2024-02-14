from typing import List, Tuple

from .Neurons import BaseNeuron, ConstNeuron, ProxyNeuron

class NeuronNetwork:
    """
    The abstract base class for a Neuron Network
    Child classes should extend `__init__`
    """
    def __init__(
        self,
        input_layer: List[ProxyNeuron],
        output_layer: List[BaseNeuron],
    ) -> None:
        """
        Child classes should use __init__ as the function to create and connect up the
        network's neurons, and call super().__init__(...) to store them as member variables here
        """
        self.input_layer = input_layer
        self.output_layer = output_layer

    def __call__(self, *inputs: int) -> Tuple[int]:
        """
        For calling the network directly with int inputs, returning int outputs.
        Useful for when evaluating a network as an ALU with binary IO.
        If chaining neuron networks together then this function should not be called;
        connect up the output layer neurons as desired instead.
        """
        assert len(inputs) == len(self.input_layer)
        valid_inputs = {0, 1}
        assert all(i in valid_inputs for i in inputs)

        float_inputs = tuple(float(i) for i in inputs) # somewhat unnecessary for Python

        cache = {} # of type `Dict[BaseNeuron, float]`

        for input_value, neuron_input in zip(float_inputs, self.input_layer):
            neuron_input.source = ConstNeuron(input_value)

        # this is a depth-first evaluation; having cache really helps
        float_outputs = tuple((neuron(cache) for neuron in self.output_layer))
        valid_float_outputs = {0.0, 1.0}

        # lest a non-integer float arises eg from a non-perceptron neuron for some reason
        assert all(o in valid_float_outputs for o in float_outputs)

        int_outputs = tuple(int(o) for o in float_outputs)

        return int_outputs
