from typing import Tuple

from libThresholdLogic import Perceptron, ProxyNeuron, NeuronNetwork

"""Boolean Logic embedded within Threshold Logic"""

class HammingGate(NeuronNetwork):
    """
    Creates a Hamming metric closed ball of radius `max_distance` around `target_vector`
    Returns 1 when input is in the closed ball, else 0
    """
    def __init__(self, target_vector: Tuple[int], max_distance: int) -> None:
        assert all(bit in {0, 1} for bit in target_vector)

        self.target_vector = target_vector
        self.max_distance = max_distance

        n_inputs = len(target_vector)

        bias = (2 * sum(target_vector) - 2 * max_distance - 1) / 2

        neuron = Perceptron(bias) # the things we can achieve with just 1 perceptron!

        input_layer = [ProxyNeuron() for _ in range(n_inputs)]

        for target_vector_bit, input_neuron in zip(target_vector, input_layer):
            weight = 1.0 if target_vector_bit else -1.0
            neuron.add_input(weight, input_neuron)

        output_layer = [neuron]

        super().__init__(input_layer, output_layer)

# XXX GAND and GNAND arguments could be styled in Latin like
# centri'petal' (seek) and centri'fugal' (flee)

class GAND(HammingGate):
    """
    Returns 1 when input is exactly `seek_vector`, else 0
    """
    def __init__(self, seek_vector: Tuple[int]) -> None:
        super().__init__(seek_vector, 0)

class GNAND(HammingGate):
    """
    Returns 1 when input is anything but `flee_vector`, else 0
    Works by using a closed ball of radius `n - 1` around
    ~flee_vector (that is the bitwise inverse)
    """
    def __init__(self, flee_vector: Tuple[int]) -> None:
        seek_vector = tuple(0 if bit else 1 for bit in flee_vector) # invert `flee_vector`
        max_distance = len(seek_vector) - 1 # every vector but `flee_vector`
        super().__init__(seek_vector, max_distance)

class AND(GAND):
    """
    Returns 1 when all inputs are 1, else 0
    """
    def __init__(self, n_inputs: int = 2) -> None:
        seek_vector = tuple(1 for _ in range(n_inputs))
        super().__init__(seek_vector)

class NOR(GAND):
    """
    Returns 1 when all inputs are 0, else 0
    """
    def __init__(self, n_inputs: int = 2) -> None:
        seek_vector = tuple(0 for _ in range(n_inputs))
        super().__init__(seek_vector)

class NAND(GNAND):
    """
    Returns 1 when any inputs is 0, else 0
    """
    def __init__(self, n_inputs: int = 2) -> None:
        flee_vector = tuple(1 for _ in range(n_inputs))
        super().__init__(flee_vector)

class OR(GNAND):
    """
    Returns 1 when any input is 1, else 0
    """
    def __init__(self, n_inputs: int = 2) -> None:
        flee_vector = tuple(0 for _ in range(n_inputs))
        super().__init__(flee_vector)

class NOT(NAND):
    """
    Returns 1 when input is 0, else 0
    """
    def __init__(self) -> None:
        super().__init__(n_inputs = 1)

class XOR(NeuronNetwork):
    """
    Returns 1 when exactly one input is 1, else 0
    """
    def __init__(self, return_carry_bit: bool = False) -> None:
        # the same result as `HalfAdder`
        neuron_sum = Perceptron(0.5)
        neuron_carry = Perceptron(1.5)

        neurons = [neuron_sum, neuron_carry]

        neuron_sum.add_input(-2.0, neuron_carry)

        input_layer = [ProxyNeuron() for _ in range(2)]

        for neuron_src in input_layer:
            neuron_sum.add_input(1.0, neuron_src)
            neuron_carry.add_input(1.0, neuron_src)

        if return_carry_bit:
            output_layer = neurons
        else:
            output_layer = neurons[:1]

        super().__init__(input_layer, output_layer)

class XNOR(NeuronNetwork):
    """
    Returns 1 when exactly zero or two inputs are 1, else 0
    """
    def __init__(self, return_carry_bit: bool = False) -> None:
        neuron_sum = Perceptron(-0.5)
        neuron_carry = Perceptron(1.5)

        neurons = [neuron_sum, neuron_carry]

        neuron_sum.add_input(2.0, neuron_carry)

        input_layer = [ProxyNeuron() for _ in range(2)]

        for neuron_src in input_layer:
            neuron_sum.add_input(-1.0, neuron_src)
            neuron_carry.add_input(1.0, neuron_src)

        if return_carry_bit:
            output_layer = neurons
        else:
            output_layer = neurons[:1]

        super().__init__(input_layer, output_layer)
