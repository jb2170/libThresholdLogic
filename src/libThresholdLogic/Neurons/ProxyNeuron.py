from typing import Optional

from .BaseNeuron import BaseNeuron

class ProxyNeuron(BaseNeuron):
    """
    A wrapper class for connecting networks' neurons up at a delayed time.
    For example the `input_layer` of a `NeuronNetwork` is comprised of `ProxyNeuron`s
    as the inputs may be either other real `Perceptron`s (linking networks together)
    or `ConstNeuron`s (for evaluating the network)
    """
    def __init__(self, source: Optional[BaseNeuron] = None) -> None:
        self.source = source

    def do_call(self, cache) -> float:
        if self.source is None:
            raise ValueError("ProxyNeuron source unset")

        return self.source(cache)
