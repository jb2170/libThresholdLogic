#!/usr/bin/env python3

from libThresholdLogic.ExampleNetworks import GenericBitAdder, int_to_bit_tuple_lb, bit_tuple_lb_to_int

def main() -> None:
    n_neurons = 3
    n_bit = 2 ** n_neurons - 1

    adder = GenericBitAdder(n_neurons)

    for i in range(2 ** n_bit):
        bits = int_to_bit_tuple_lb(i, n_bit)

        res = bit_tuple_lb_to_int(adder(*bits))

        print(bits, res)
        assert res == sum(bits)

if __name__ == "__main__":
    main()
