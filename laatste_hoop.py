import pandas as pd
import numpy as np
import math

# DEFINITIONS
# f = [a1, a2, ..., b1, b2]
# phi = [a1, a2, b1, b2, l3, d]

### Mathematical functions
def calc_S(q, x, y, l1, l2, l3):
    if q == 1 and l3 == 0:
        return 0
    if q == 0 and min(x, y) == 0:
        return 1

    total = 0
    for k in range(min(x, y) + 1):
        total += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (k**q) * ((l3)/(l1*l2))**k

    return total


def calc_U(x, y, l1, l2, l3):
    return calc_S(1, x, y, l1, l2, l3) / calc_S(0, x, y, l1, l2, l3)


def calc_Delta_ijt(x, y, l1, l2, l3):
    U = calc_U(x, y, l1, l2, l3)
    return np.array([
        x - l1 - U,
        y - l2 - U,
        l2 - y + U,
        l1 - x + U
    ])

print(calc_Delta_ijt(3, 4, .5, .5, .5))