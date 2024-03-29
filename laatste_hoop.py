import pandas as pd
import numpy as np
import math
import time as tm
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("error")

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
        last_term = 0 if l3 == 0 else l3/(l1*l2)
        total += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (k**q) * last_term**k

    return total


def calc_Pbp(x, y, l1, l2, l3):
    e = np.exp(-1 * (l1 + l2 + l3))
    f = ((l1**x) / (math.factorial(x))) * ((l2**y) / (math.factorial(y)))

    return e * f * calc_S(0, x, y, l1, l2, l3)


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
            x, y = int(match[0]), int(match[1])

            round_ll += np.log(calc_Pbp(x, y, l1, l2, l3))
            f_update = update_f_ijt(f_ij, w_ij, A, B, x, y, l1, l2, l3)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

        for n in np.where(teams_played == False):
            f[n] = w[n] + b1 * f[n]
            f[n+N] = w[n+N] + b2 * f[n+N]

        likelihood += round_ll

    return -1 * likelihood


def forecast_f(params, init_f, data):
    a1, a2, b1, b2, l3, d = params
    f = np.copy(init_f)
    N = len(f) // 2
    w = np.multiply(f, np.concatenate((np.repeat(1-b1, N), np.repeat(1-b2, N))))
    A = np.diag([a1, a1, a2, a2])
    B = np.diag([b1, b1, b2, b2])

    rounds = np.unique(data[:,6])
    ajax = np.zeros(len(rounds))
    fey = np.zeros(len(rounds))
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j, i+N, j+N]
            f_ij = np.array([f[i] for i in indices])
            w_ij = np.array([w[i] for i in indices])
            l1 = np.exp(d + f_ij[0] - f_ij[3])
            l2 = np.exp(f_ij[1] - f_ij[2])
            x, y = int(match[0]), int(match[1])

            f_update = update_f_ijt(f_ij, w_ij, A, B, x, y, l1, l2, l3)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

        for n in np.where(teams_played == False):
            f[n] = w[n] + b1 * f[n]
            f[n+N] = w[n+N] + b2 * f[n+N]

        # print(f"Ajax: {f[6]}, Feyenoord: {f[14]}")
        ajax[r - rounds[0]] = f[6]
        fey[r - rounds[0]] = f[14]

    sns.lineplot(y=ajax, x=rounds, color='red')
    sns.lineplot(y=fey, x=rounds)
    plt.show()
    return f

def main():
    # FTHG  FTAG    FTR Season  HomeTeamID  AwayTeamID  RoundNO
    # 0     1       2   3       4           5           6
    data = pd.read_csv("processed_data.csv", usecols=["Season", "RoundNO", "HomeTeamID", "AwayTeamID", "FTR", "FTHG", "FTAG"])
    ftr_mapping = {'A': 0, 'D': 1, 'H': 2}
    data['FTR'] = data['FTR'].apply(lambda x: ftr_mapping[x])

    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 16
    usable_data = data[data.Season > 0]
    train_data = usable_data[usable_data.Season <= split_season].to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)

    N = max(np.max(data.HomeTeamID), np.max(data.AwayTeamID)) + 1
    Sx = np.sum(init_data[:,0]) + np.sum(init_data[:, 1])
    a_init = np.zeros(18)
    b_init = np.zeros(18)
    for i in np.unique(init_data[:,4]):
        a_init[i] = np.sum(init_data[init_data[:, 4] == i][:, 0]) / np.sqrt(Sx)
        b_init[i] = np.sum(init_data[init_data[:, 5] == i][:, 0]) / np.sqrt(Sx)
    
    a_init = a_init - np.mean(a_init)
    b_init = b_init - np.mean(b_init)
    f_init = np.concatenate((a_init, np.zeros(N - len(a_init)), b_init, np.zeros(N - len(b_init))))
    print(f_init)
    print(f"STARTING: Ajax: {f_init[6]}, Feyenoord: {f_init[14]}")

    x0 = (0.1, 0.1, 0.1, 0.1, 0.5, 0.5)
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (0, 5), (0, 5)]
    result = minimize(calculate_ll, x0, args=(f_init, init_data), method='Nelder-Mead', bounds=bounds, options={'maxiter': 100})
    print(result)
    f = forecast_f(result.x, f_init, init_data)
    print(f"AFTER 1 year: Ajax: {f[6]}, Feyenoord: {f[14]}")
    # params = minimize(calculate_ll, x0, args=(f_init, train_data), method='Nelder-Mead', bounds=bounds, options={'maxiter': 300})
    # print(params)
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