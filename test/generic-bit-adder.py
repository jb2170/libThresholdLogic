#!/usr/bin/env python3

from libThresholdLogic.ExampleNetworks import GenericBitAdder

def main() -> None:
    adder = GenericBitAdder(3)
    print(adder(1, 0, 0, 0, 0, 0, 0))

    n_bits = 2 ** 3 - 1

    for i in range(2 ** n_bits):
        bits = tuple(int(c) for c in f"{i:0{n_bits}b}")
        print(sum(bits), tuple(int(c) for c in adder(*bits)))

if __name__ == "__main__":
    main()
