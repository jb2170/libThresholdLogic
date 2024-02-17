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

class FullAdder(NeuronNetwork):
    def __init__(self) -> None:
        neuron_sum = Perceptron(1.0)
        neuron_carry = Perceptron(1.0)

        neurons = [neuron_sum, neuron_carry]

        # inhibitory connection from carry to sum
        neuron_sum.add_input(-2.0, neuron_carry)

        input_layer = [ProxyNeuron() for _ in range(3)]

        # excitatory connections from inputs
        for neuron_src in input_layer:
            neuron_sum.add_input(1.0, neuron_src)
            neuron_carry.add_input(0.5, neuron_src)

        output_layer = neurons

        super().__init__(input_layer, output_layer)

class GenericBitAdder(NeuronNetwork):
    """
    When `n_neurons` is `2` we recover a `FullAdder`
    """
    def __init__(self, n_neurons: int) -> None:
        self.n_neurons = n_neurons
        self.n_inputs = 2 ** n_neurons - 1

        neurons = [Perceptron(1.0) for _ in range(n_neurons)]

        # inhibitory connections between neurons
        for neuron_src_idx, neuron_src in enumerate(neurons):
            for neuron_dest_idx, neuron_dest in enumerate(neurons[:neuron_src_idx]):
                neuron_dest.add_input(-(2 ** (neuron_src_idx - neuron_dest_idx)), neuron_src)

        input_layer = [ProxyNeuron() for _ in range(self.n_inputs)]

        # excitatory connections from inputs
        for neuron_src_idx, neuron_src in enumerate(input_layer):
            for neuron_dest_idx, neuron_dest in enumerate(neurons):
                neuron_dest.add_input(2 ** (-neuron_dest_idx), neuron_src)

        output_layer = neurons

        # XXX perhaps a more formal wrapper class `ProxyNeuronSet` could be useful?
        self.carry_inputs  = input_layer[:n_neurons - 1] # of which there are n_neurons - 1
        self.real_inputs   = input_layer[n_neurons - 1:] # of which there are 2 ** n_neurons - n_neurons
        self.real_output   = output_layer[0]             # of which there is 1, returned by value, not in a list
        self.carry_outputs = output_layer[1:]            # of which there are n_neurons - 1

        super().__init__(input_layer, output_layer)

class GenericNumberAdder(NeuronNetwork):
    # TODO Thesis figure 7.4 diagram wiring is incorrect (code is correct however)
    """
    Uses `GenericBitAdder`s in a ripple carry fashion which generalises the ripple
    carry possible with `FullAdder`s.
    Adds together `(2 ** n_neurons) - n_neurons` lots of `n_bit` numbers
    """
    def __init__(self, n_bit: int, n_neurons: int) -> None:
        self.n_bit = n_bit
        self.n_neurons = n_neurons

        bit_adders = [GenericBitAdder(n_neurons) for _ in range(n_bit)]

        bit_adder_n_real_inputs = len(bit_adders[0].real_inputs)
        bit_adder_n_carry_inputs = len(bit_adders[0].carry_inputs)

        # arrange input layer such that the network accepts a sequence of little-bittian numbers
        input_layer = []
        for _ in range(bit_adder_n_real_inputs):
            input_layer_number = [ProxyNeuron() for _ in range(n_bit)]
            for neuron, bit_adder in zip(input_layer_number, bit_adders):
                bit_adder.connect_inputs(neuron)
            input_layer += input_layer_number

        output_layer = [bit_adder.real_output for bit_adder in bit_adders]

        # connect carry-out neurons to the carry-in neurons of the successive bit-adder units
        for carry_out_adder_idx, carry_out_adder in enumerate(bit_adders):
            carry_in_adders = bit_adders[carry_out_adder_idx + 1:]
            for carry_out_neuron, carry_in_adder in zip(carry_out_adder.carry_outputs, carry_in_adders):
                carry_in_adder.connect_inputs(carry_out_neuron)

        # for the first `bit_adder_n_carry_inputs` bit adders indexed by `j`,
        # the last `bit_adder_n_carry_inputs - j` carry inputs need to be tied to 0
        # as there is nothing carrying into them
        for carry_in_adder in bit_adders[:bit_adder_n_carry_inputs]:
            carry_in_adder.pad_unconnected_inputs()

        super().__init__(input_layer, output_layer)
