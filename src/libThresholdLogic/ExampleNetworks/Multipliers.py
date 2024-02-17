from typing import List, Tuple
import math

from libThresholdLogic import ProxyNeuron, NeuronNetwork
from .Adders import GenericBitAdder
from .LogicGates import AND, GAND, XOR

class BitMultiplier2x2(NeuronNetwork):
    """2x2 bit multiplication done using only 6 perceptrons"""

    def __init__(self) -> None:
        _1s_and = AND()

        _2s_and1 = AND()
        _2s_and2 = AND()
        _2s_xor = XOR(return_carry_bit = True)

        _4s_gand = GAND((1, 1, 0))

        x0 = ProxyNeuron()
        x1 = ProxyNeuron()
        y0 = ProxyNeuron()
        y1 = ProxyNeuron()

        _1s_and.connect_inputs(x0, y0)

        _2s_and1.connect_inputs(x1, y0)
        _2s_and2.connect_inputs(x0, y1)
        _2s_xor.connect_inputs(_2s_and1.output_layer[0], _2s_and2.output_layer[0])
        _4s_gand.connect_inputs(x1, y1, _1s_and.output_layer[0])

        input_layer = [x0, x1, y0, y1]
        output_layer = [
            _1s_and.output_layer[0],  # 1s
            _2s_xor.output_layer[0],  # 2s
            _4s_gand.output_layer[0], # 4s
            _2s_xor.output_layer[1]   # 8s
        ]

        super().__init__(input_layer, output_layer)

class GenericBitMultiplier(NeuronNetwork):
    @staticmethod
    def n_neurons_per_adder(n_bit: int) -> List[int]:
        """
        Returns a list of the number of neurons needed in our generic bit adders
        for `n_bit` multiplication with all carries accounted for
        """

        n_inputs = [t + 1 for t in range(n_bit)] + [n_bit - 1 - t for t in range(n_bit - 1)]

        idx = 0
        while True:
            try:
                n_inputs_idx = n_inputs[idx]
            except IndexError:
                # end of n_inputs
                break

            n_carry_bits = math.ceil(math.log2(n_inputs_idx + 1)) - 1

            need_length = idx + 1 + n_carry_bits
            n_inputs += [0] * (need_length - len(n_inputs))
            # pad n_inputs with zeros if there's more carry bits extending beyond the original length of n_inputs

            for j in range(n_carry_bits):
                n_inputs[idx + 1 + j] += 1

            idx += 1

        n_neurons = [math.ceil(math.log2(i + 1)) for i in n_inputs]

        return n_neurons

    @staticmethod
    def convolution_indices(n: int) -> List[List[Tuple[int, int]]]:
        return [
            [
                (u, t - u) for u in range(t + 1)
            ]
            for t in range(n)
        ] + [
            [
                (t + u, n - 1 - u) for u in range(n - t)
            ]
            for t in range(1, n)
        ]

    def __init__(self, n_bit: int) -> None:
        self.n_bit = n_bit

        input_layer = [ProxyNeuron() for _ in range(2 * n_bit)]
        input_neurons_x = input_layer[:n_bit]
        input_neurons_y = input_layer[n_bit:]

        adders = [GenericBitAdder(n) for n in self.n_neurons_per_adder(n_bit)]

        convolution_indices = self.convolution_indices(n_bit)

        for idx, (adder, convol_indices) in enumerate(zip(adders, convolution_indices)):
            # there is often less convol_indices than adders
            # the last adders are purely for carry bits
            for input_neurons_x_idx, input_neurons_y_idx in convol_indices:
                and_gate = AND()
                and_gate.connect_inputs(input_neurons_x[input_neurons_x_idx], input_neurons_y[input_neurons_y_idx])
                adder.connect_inputs(and_gate.output_layer[0])

            adder.pad_unconnected_inputs()

            for carry_adder_idx, carry_bit_neuron in enumerate(adder.carry_outputs, start = idx + 1):
                adders[carry_adder_idx].connect_inputs(carry_bit_neuron)

        for adder in adders[len(convol_indices):]:
            adder.pad_unconnected_inputs()

        output_layer = [adder.real_output for adder in adders]

        super().__init__(input_layer, output_layer)
