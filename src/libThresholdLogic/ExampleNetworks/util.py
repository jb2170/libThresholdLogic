from typing import Tuple

# 'bittian' inspired by 'endian'ness
# enumerating the bits of a binary number in little bittian form corresponds to
# 1s, 2s, 4s, 8s, ...
# and big bittian is in reverse

def int_to_bit_tuple_lb(x: int, n_bits: int) -> Tuple[int]:
    """First `n_bits` of x in little bittian"""
    return tuple((
        1 if x & (1 << bit_n) else 0
        for bit_n in range(n_bits)
    ))

def int_to_bit_tuple_bb(x: int, n_bits: int) -> Tuple[int]:
    """First `n_bits` of x in big bittian"""
    return int_to_bit_tuple_lb(x, n_bits)[::-1]

def bit_tuple_lb_to_int(t: Tuple[int]) -> int:
    """Little bittian tuple to int"""
    return sum((
        2 ** idx if bit else 0
        for (idx, bit) in enumerate(t)
    ))

def bit_tuple_bb_to_int(t: Tuple[int]) -> int:
    """Big bittian tuple to int"""
    return bit_tuple_lb_to_int(t[::-1])
