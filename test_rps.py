import numpy as np

def calculate_RPS(realizations, probabilities):
    rps = np.zeros(len(realizations))
    for i, (o, p) in enumerate(zip(realizations, probabilities)):
        cdf = np.ones(3)
        if int(o) != 0:
            cdf[0] = 0
        if int(o) == 2:
            cdf[1] = 0
        
        squares = np.square([p[0] - cdf[0], p[0] + p[1] - cdf[1], 0])
        rps[i] = np.sum(squares)/2

    return rps

x = calculate_RPS([2], [[0.2, 0.3, 0.5]])
print(x)