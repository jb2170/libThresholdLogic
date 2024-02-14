#!/usr/bin/env python3

from libThresholdLogic.ExampleNetworks import HalfAdder, int_to_bit_tuple_lb, bit_tuple_lb_to_int

def main() -> None:
    n_bit = 1
    adder = HalfAdder()
    for y in range(2 ** n_bit):
        y_bits = int_to_bit_tuple_lb(y, n_bit)
        for x in range(2 ** n_bit):
            x_bits = int_to_bit_tuple_lb(x, n_bit)

            res = bit_tuple_lb_to_int(adder(*(x_bits + y_bits)))

            print(f"{x = }, {y = }, x + y = {res}")
            assert x + y == res

if __name__ == "__main__":
    main()
