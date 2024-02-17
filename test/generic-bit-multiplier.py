#!/usr/bin/env python3

from libThresholdLogic.ExampleNetworks import GenericBitMultiplier, int_to_bit_tuple_lb, bit_tuple_lb_to_int

def main() -> None:
    n_bit = 8
    mult = GenericBitMultiplier(n_bit)
    for y in range(2 ** n_bit):
        y_bits = int_to_bit_tuple_lb(y, n_bit)
        for x in range(2 ** n_bit):
            x_bits = int_to_bit_tuple_lb(x, n_bit)

            res = bit_tuple_lb_to_int(mult(*(x_bits + y_bits)))

            print(x, y, res)
            assert x * y == res

if __name__ == "__main__":
    main()
