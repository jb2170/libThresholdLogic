from libThresholdLogic import Perceptron, ProxyNeuron, NeuronNetwork

class HalfAdder(NeuronNetwork):
    def __init__(self) -> None:
        neuron_sum = Perceptron(1.0)
        neuron_carry = Perceptron(1.0)

        neurons = [neuron_sum, neuron_carry]

        # inhibitory connection from carry to sum
        neuron_sum.add_input(-2.0, neuron_carry)

        input_layer = [ProxyNeuron() for _ in range(2)]

        # excitatory connections from inputs
        for neuron_src in input_layer:
            neuron_sum.add_input(1.0, neuron_src)
            neuron_carry.add_input(0.5, neuron_src)

        output_layer = neurons

        super().__init__(input_layer, output_layer)
