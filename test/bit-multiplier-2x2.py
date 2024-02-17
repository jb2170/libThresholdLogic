#!/usr/bin/env python3

from typing import List

from libThresholdLogic.ExampleNetworks import BitMultiplier2x2, int_to_bit_tuple_lb, bit_tuple_lb_to_int

def add_ints(adder, inputs: List[int]):
    inputs = sum([
        int_to_bit_tuple_lb(input_, adder.n_bit_input)
        for input_ in inputs
    ], tuple())
    print(inputs)
    return bit_tuple_lb_to_int(adder(*inputs))

def main() -> None:
    mult = BitMultiplier2x2()
    for y in range(4):
        y_bits = int_to_bit_tuple_lb(y, 2)
        for x in range(4):
            x_bits = int_to_bit_tuple_lb(x, 2)

            res = bit_tuple_lb_to_int(mult(*(x_bits + y_bits)))

            assert x * y == res
            print(x, y, res)

if __name__ == "__main__":
    main()
