import pandas as pd
import numpy as np
import math
import time as tm

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


def update_f_ijt(f, w, A, B, x, y, l1, l2, l3):
    s_ij = calc_Delta_ijt(x, y, l1, l2, l3)

    return w + np.matmul(B, f) + np.matmul(A, s_ij)


def calculate_ll(a1, a2, b1, b2, l3, d, init_f, data):
    likelihood = 0
    f = np.copy(init_f)
    N = len(f) // 2
    w = np.multiply(f, np.concatenate((np.repeat(1-b1, N), np.repeat(1-b2, N))))
    A = np.diag([a1, a2])
    B = np.diag([b1, b2])

    rounds = np.unique(data[:,6])
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]
        print(r_data)

        for match in r_data:
            i, j = match[4], match[5]
            indices = [i, j, i+N, j+N]
            f_ij = np.array([f[i] for i in indices])
            print(i, j)
            # f_update = update_f_ijt()

def main():
    # FTHG  FTAG    FTR Season  HomeTeamID  AwayTeamID  RoundNO
    # 0     1       2   3       4           5           6
    data = pd.read_csv("processed_data.csv", usecols=["Season", "RoundNO", "HomeTeamID", "AwayTeamID", "FTR", "FTHG", "FTAG"]).head(500)
    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 2
    usable_data = data[data.Season > 0]
    train_data = usable_data[usable_data.Season <= split_season].to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)

    N = max(np.max(data.HomeTeamID), np.max(data.AwayTeamID)) + 1
    f_init = np.repeat(0.4, 2*N)
    calculate_ll(0.1, 0.2, 0.1, 0.2, 0.5, 0.2, f_init, train_data)

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