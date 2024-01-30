import pandas as pd
import numpy as np
import math
import time as tm
from scipy.optimize import minimize

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

    return total + 1e-10


def calc_Pbp(x, y, l1, l2, l3):
    e = np.exp(-1 * (l1 + l2 + l3))
    f = ((l1**x) / (math.factorial(x))) * ((l2**y) / (math.factorial(y)))

    return e * f * calc_S(0, x, y, l1, l2, l3) + 1e-10


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


def update_f_ijt(f, w, A, B, x, y, l1, l2, l3):
    s_ij = calc_Delta_ijt(x, y, l1, l2, l3)

    return w + np.matmul(B, f) + np.matmul(A, s_ij)


def calculate_ll(params, init_f, data):
    a1, a2, b1, b2, l3, d = params
    likelihood = 0
    f = np.copy(init_f)
    N = len(f) // 2
    w = np.multiply(f, np.concatenate((np.repeat(1-b1, N), np.repeat(1-b2, N))))
    A = np.diag([a1, a1, a2, a2])
    B = np.diag([b1, b1, b2, b2])

    rounds = np.unique(data[:,6])
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]
        round_ll = 0

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j, i+N, j+N]
            f_ij = np.array([f[i] for i in indices])
            w_ij = np.array([w[i] for i in indices])
            l1 = np.exp(d + f_ij[0] - f_ij[3])
            l2 = np.exp(f_ij[1] - f_ij[2])
            print(l1, l2)
            x, y = int(match[0]), int(match[1])

            round_ll += calc_Pbp(x, y, l1, l2, l3)
            f_update = update_f_ijt(f_ij, w_ij, A, B, x, y, l1, l2, l3)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

        for n in np.where(teams_played == False):
            f[n] = w[n] + b1 * f[n]
            f[n+N] = w[n+N] + b2 * f[n+N]

        likelihood += np.log(round_ll)
    print(likelihood)
    return -1 * likelihood

def main():
    # FTHG  FTAG    FTR Season  HomeTeamID  AwayTeamID  RoundNO
    # 0     1       2   3       4           5           6
    data = pd.read_csv("processed_data.csv", usecols=["Season", "RoundNO", "HomeTeamID", "AwayTeamID", "FTR", "FTHG", "FTAG"])
    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 5
    usable_data = data[data.Season > 0]
    train_data = usable_data[usable_data.Season <= split_season].to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)

    N = max(np.max(data.HomeTeamID), np.max(data.AwayTeamID)) + 1
    f_init = np.repeat(2, 2*N)
    x0 = (0.5, 0.5, 0.5, 0.5, 0.5, 2)
    params = minimize(calculate_ll, x0, args=(f_init, train_data))
    print(params)
    # calculate_ll(params, f_init, train_data)



if __name__ == '__main__':
    main()


    # X = np.diagflat(np.concatenate((np.ones(4), np.repeat(2, 4))))
    # M = np.zeros((4, 8))
    # M[0, 1] = 1
    # M[1, 6] = 1
    # M[2, 7] = 1
    # M[3, 4] = 1
    # print(np.matmul(np.matmul(M, X), M.T))
# f_ij = np.array([f[i] for i in indexes_needed])