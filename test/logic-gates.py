#!/usr/bin/env python3

from libThresholdLogic.ExampleNetworks import HammingGate, GAND, GNAND, AND, NOR, NAND, OR, NOT, XOR, XNOR, int_to_bit_tuple_lb

def expected_hamming(gate: HammingGate, *inputs: int) -> int:
    assert len(inputs) == len(gate.input_layer)

    distance = sum(
        0 if bit_target == bit_input else 1
        for bit_target, bit_input in zip(gate.target_vector, inputs)
    )

    return int(distance <= gate.max_distance)

def expected_gand(gate: GAND, *inputs):
    assert len(inputs) == len(gate.target_vector)
    return int(all(i == t for i, t in zip(inputs, gate.target_vector)))

def expected_gnand(gate: GNAND, *inputs):
    assert len(inputs) == len(gate.target_vector)
    target_vector = tuple(0 if bit else 1 for bit in gate.target_vector) # invert flee_vector
    return int(not all(i == t for i, t in zip(inputs, target_vector)))

def expected_and(*inputs: int) -> int:
    return int(all(inputs))

def expected_nor(*inputs: int) -> int:
    return int(not any(inputs))

def expected_or(*inputs: int) -> int:
    return int(any(inputs))

def expected_nand(*inputs: int) -> int:
    return int(not all(inputs))

def expected_not(input_: int) -> int:
    return int(not input_)

def expected_xor(input_1: int, input_2: int) -> int:
    return int(input_1 ^ input_2)

def expected_xnor(input_1: int, input_2: int) -> int:
    return int(not (input_1 ^ input_2))

def test_linears() -> None:
    n_inputs = 3
    for threshold_gate_t, actual_gate in (
        (AND, expected_and),
        (NOR, expected_nor),
        (NAND, expected_nand),
        (OR, expected_or),
    ):
        print(threshold_gate_t.__name__)
        threshold_gate = threshold_gate_t(n_inputs)
        for i in range(2 ** n_inputs):
            input_bits = int_to_bit_tuple_lb(i, n_inputs)
            gate_output, = threshold_gate(*input_bits)
            print(input_bits, gate_output)
            expected_output = actual_gate(*input_bits)
            assert gate_output == expected_output
        print()

def test_not() -> None:
    n_inputs = 1
    threshold_gate_t, actual_gate = NOT, expected_not
    print(threshold_gate_t.__name__)
    threshold_gate = threshold_gate_t()
    for i in range(2 ** n_inputs):
        input_bits = int_to_bit_tuple_lb(i, n_inputs)
        gate_output, = threshold_gate(*input_bits)
        print(input_bits, gate_output)
        expected_output = actual_gate(*input_bits)
        assert gate_output == expected_output
    print()

def test_hamming() -> None:
    target_vector = (1, 1, 0, 0, 1)
    max_distance = 2
    threshold_gate_t, actual_gate = HammingGate, expected_hamming
    print(threshold_gate_t.__name__)
    threshold_gate = threshold_gate_t(target_vector, max_distance)
    n_inputs = len(target_vector)
    for i in range(2 ** n_inputs):
        input_bits = int_to_bit_tuple_lb(i, n_inputs)
        gate_output, = threshold_gate(*input_bits)
        print(input_bits, gate_output)
        expected_output = actual_gate(threshold_gate, *input_bits)
        assert gate_output == expected_output
    print()

def test_gand() -> None:
    seek_vector = (1, 1, 0, 0, 1)
    threshold_gate_t, actual_gate = GAND, expected_gand
    print(threshold_gate_t.__name__)
    threshold_gate = threshold_gate_t(seek_vector)
    n_inputs = len(seek_vector)
    for i in range(2 ** n_inputs):
        input_bits = int_to_bit_tuple_lb(i, n_inputs)
        gate_output, = threshold_gate(*input_bits)
        print(input_bits, gate_output)
        expected_output = actual_gate(threshold_gate, *input_bits)
        assert gate_output == expected_output
    print()

def test_gnand() -> None:
    flee_vector = (1, 1, 0, 0, 1)
    threshold_gate_t, actual_gate = GNAND, expected_gnand
    print(threshold_gate_t.__name__)
    threshold_gate = threshold_gate_t(flee_vector)
    n_inputs = len(flee_vector)
    for i in range(2 ** n_inputs):
        input_bits = int_to_bit_tuple_lb(i, n_inputs)
        gate_output, = threshold_gate(*input_bits)
        print(input_bits, gate_output)
        expected_output = actual_gate(threshold_gate, *input_bits)
        assert gate_output == expected_output
    print()

def test_exclusives() -> None:
    n_inputs = 2
    for threshold_gate_t, actual_gate in (
        (XOR, expected_xor),
        (XNOR, expected_xnor),
    ):
        print(threshold_gate_t.__name__)
        threshold_gate = threshold_gate_t()
        for i in range(2 ** n_inputs):
            input_bits = int_to_bit_tuple_lb(i, n_inputs)
            gate_output, = threshold_gate(*input_bits)
            print(input_bits, gate_output)
            expected_output = actual_gate(*input_bits)
            assert gate_output == expected_output
        print()

def main() -> None:
    test_linears()
    test_not()
    test_hamming()
    test_gand()
    test_gnand()
    test_exclusives()

if __name__ == "__main__":
    main()
