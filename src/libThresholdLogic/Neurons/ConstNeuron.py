from .BaseNeuron import BaseNeuron

class ConstNeuron(BaseNeuron):
    """
    A neuron that always returns a constant
    """
    def __init__(self, value: float) -> None:
        self.value = value

    def do_call(self, cache) -> float:
        return self.value
