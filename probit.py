import pandas as pd
import numpy as np
import math
import time as tm
from scipy.optimize import minimize
import scipy.stats as sts
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("error")

# DEFINITIONS
# f = [a1, a2, ..., b1, b2]
# phi = [a1, a2, b1, b2, l3, d]

### Mathematical functions
def calc_Pbp(C, l6, k1, k2):
    if C == 2:
        return sts.norm.cdf(k1 - l6)
    if C == 1:
        return sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6)
    if C == 0:
        return 1 - sts.norm.cdf(k2 - l6)


def calc_Delta_ijt(C, l6, k1, k2):
    if C == 2:
        elem_1 = sts.norm.pdf(k2 - l6) / (1 - sts.norm.cdf(k2 - l6))
        elem_2 = -1*sts.norm.pdf(k2 - l6) / (1 - sts.norm.cdf(k2 - l6))
        return np.array([elem_1, elem_2])
    if C == 1:
        elem_1 = (sts.norm.pdf(k1 - l6) - sts.norm.pdf(k2 - l6)) / (sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6))
        elem_2 = (sts.norm.pdf(k2 - l6) - sts.norm.pdf(k1 - l6)) / (sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6))
        return np.array([elem_1, elem_2])
    if C == 0:
        elem_1 = -1*sts.norm.pdf(k1 - l6) / (1 - sts.norm.cdf(k1 - l6))
        elem_2 = sts.norm.pdf(k1 - l6) / (1 - sts.norm.cdf(k1 - l6))
        return np.array([elem_1, elem_2])


def update_f_ijt(f, w, A, B, C, l6, k1, k2):
    s_ij = calc_Delta_ijt(C, l6, k1, k2)

    return w + np.matmul(B, f) + np.matmul(A, s_ij)


def calculate_ll(params, init_f, data):
    a1, b1, k1, k2star = params
    k2 = k1 + np.exp(k2star)
    likelihood = 0
    f = np.copy(init_f)
    N = len(f)
    w = np.multiply(f, np.repeat(1-b1, N))
    A = np.diag([a1, a1])
    B = np.diag([b1, b1])

    rounds = np.unique(data[:,6])
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]
        round_ll = 0

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j]
            f_ij = np.array([f[i] for i in indices])
            w_ij = np.array([w[i] for i in indices])
            
            C = match[2]
            l6 = f_ij[0] - f_ij[1]

            round_ll += np.log(calc_Pbp(C, l6, k1, k2))
            f_update = update_f_ijt(f_ij, w_ij, A, B, C, l6, k1, k2)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

        # for n in np.where(teams_played == False):
        #     f[n] = w[n] + b1 * f[n]
        #     f[n+N] = w[n+N] + b2 * f[n+N]

        likelihood += round_ll

    print(likelihood)
    return -1 * likelihood


def calculate_static_ll(params, data):
    k1 = params[0]
    k2 = params[0] + np.exp(params[1])
    likelihood = 0
    f = np.array(params[2:])
    N = len(f)

    rounds = np.unique(data[:,6])
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]
        round_ll = 0

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j]
            f_ij = np.array([f[i] for i in indices])
            
            C = match[2]
            l6 = f_ij[0] - f_ij[1]

            round_ll += np.log(calc_Pbp(C, l6, k1, k2))

        # for n in np.where(teams_played == False):
        #     f[n] = w[n] + b1 * f[n]
        #     f[n+N] = w[n+N] + b2 * f[n+N]

        likelihood += round_ll

    print(likelihood)
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
    N = np.max(data['HomeTeamID'])

    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 2
    usable_data = data[data.Season > 0]
    train_data = usable_data[usable_data.Season <= split_season].to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)

    ESTIMATE_INITIAL_GAMMA = False
    if ESTIMATE_INITIAL_GAMMA:
        k_start = np.array([1, 1])
        gamma_start = np.repeat(0.5, 18)
        x0 = np.concatenate((k_start, gamma_start))
        result = minimize(calculate_static_ll, x0, args=(init_data), method='Nelder-Mead', tol=1e-4)
        print(result)
        output = pd.DataFrame()
        output["Value"] = result.x
        output.to_csv("initial_gamma_probit.csv")
    else:
        df = pd.read_csv("initial_gamma_probit.csv")
        result = np.array(df["Value"])

    ESTIMATE_OVER_ALL_DATA = True
    if ESTIMATE_OVER_ALL_DATA:
        x0 = np.array([0.5, 0.5, result[0], result[1]])
        gamma_init = np.concatenate((np.array(result[2:-1]), np.zeros(N - 18)))
        all_data_result = minimize(calculate_ll, x0, args=(gamma_init, train_data), method='Nelder-Mead', tol=1e-4)
        print(all_data_result)
        output = pd.DataFrame()
        output["Value"] = all_data_result.x
        output.to_csv("full_probit_estimates.csv")

    
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