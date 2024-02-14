from libThresholdLogic import ProxyNeuron, NeuronNetwork, Perceptron

class GenericBitAdder(NeuronNetwork):
    """
    When `n_neurons` is `2` we recover a `FullAdder`
    """
    def __init__(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons

        neurons = [Perceptron(1.0) for _ in range(n_neurons)]

        # inhibitory connections between neurons
        for neuron_src_idx, neuron_src in enumerate(neurons):
            for neuron_dest_idx, neuron_dest in enumerate(neurons[:neuron_src_idx]):
                neuron_dest.add_input(-(2 ** (neuron_src_idx - neuron_dest_idx)), neuron_src)

        n_inputs = 2 ** n_neurons - 1

        input_layer = [ProxyNeuron() for _ in range(n_inputs)]

        # excitatory connections from inputs
        for neuron_src_idx, neuron_src in enumerate(input_layer):
            for neuron_dest_idx, neuron_dest in enumerate(neurons):
                neuron_dest.add_input(2 ** (-neuron_dest_idx), neuron_src)

        output_layer = neurons

        # XXX perhaps a more formal wrapper class `ProxyNeuronSet` could be useful?
        self.carry_inputs = input_layer[:n_neurons - 1] # of which there are n_neurons - 1
        self.real_inputs = input_layer[n_neurons - 1:]  # of which there are 2 ** n_neurons - n_neurons
        self.real_output = output_layer[0]              # of which there is 1, returned by value, not in a list
        self.carry_outputs = output_layer[1:]           # of which there are n_neurons - 1

        super().__init__(input_layer, output_layer)
