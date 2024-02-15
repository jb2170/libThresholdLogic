#!/usr/bin/env python3

from typing import Tuple
import itertools

from libThresholdLogic.ExampleNetworks import GenericNumberAdder, int_to_bit_tuple_lb, bit_tuple_lb_to_int

def add_ints(adder, inputs: Tuple[int]):
    input_bits = sum((
        int_to_bit_tuple_lb(input_, adder.n_bit)
        for input_ in inputs
    ), tuple()) # concatenate little-bittian tuples
    return bit_tuple_lb_to_int(adder(*input_bits))

def main_slow() -> None:
    adder = GenericNumberAdder(8, 3)
    up_to = range(16)
    for nums in itertools.product(up_to, up_to, up_to, up_to, up_to):
        res = add_ints(adder, nums)
        expected = sum(nums)
        print(nums, res)
        assert res == expected

def main() -> None:
    adder = GenericNumberAdder(32, 3)
    nums = (19_374, 828_383, 77_312, 291_382, 99)
    res = add_ints(adder, nums)
    expected = sum(nums)
    print(f"sum of {nums} = {res:_}")
    assert res == expected

if __name__ == "__main__":
    main()
