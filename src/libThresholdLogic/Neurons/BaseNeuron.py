class BaseNeuron:
    """
    The abstract base class for all neuron-like components of a Neuron Network
    Child classes must override `do_call`
    """
    def do_call(self, cache) -> float:
        """
        The implementation of evaluating the neuron
        """
        raise NotImplementedError

    def __call__(self, cache) -> float:
        """
        Evaluate the neuron's value and store it in cache, a dictionary of type `Dict[BaseNeuron, float]`
        """
        if self in cache:
            return cache[self]
        else:
            ret = self.do_call(cache)
            cache[self] = ret
            return ret
