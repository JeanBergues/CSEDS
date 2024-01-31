import pandas as pd
import numpy as np
import math
import scipy.optimize as opt
import scipy.stats as sts
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
        last_term = 0 if l3 == 0 else l3/(l1*l2 + 1e-12)
        total += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (k**q) * last_term**k

    return total


def calc_Pbp(x, y, l1, l2, l3):
    e = np.exp(-1 * (l1 + l2 + l3))
    f = ((l1**x) / (math.factorial(x))) * ((l2**y) / (math.factorial(y)))

    return e * f * calc_S(0, x, y, l1, l2, l3) + 1e-12


def calc_U(x, y, l1, l2, l3):
    return calc_S(1, x, y, l1, l2, l3) / (calc_S(0, x, y, l1, l2, l3) + 1e-12)


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
            f_ij = np.array([f[i] if np.abs(f[i]) < 3 else 3 * np.sign(f[i]) for i in indices])
            w_ij = np.array([w[i] for i in indices])
            l1 = np.exp(d + f_ij[0] - f_ij[3])
            l2 = np.exp(f_ij[1] - f_ij[2])
            x, y = int(match[0]), int(match[1])

            round_ll += np.log(calc_Pbp(x, y, l1, l2, l3))
            f_update = update_f_ijt(f_ij, w_ij, A, B, x, y, l1, l2, l3)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

        for n in np.where(teams_played == False):
            if f[n][0] != 0:
                f[n] = w[n] + b1 * f[n]
                f[n+N] = w[n+N] + b2 * f[n+N]

        likelihood += round_ll

    print(likelihood)
    return -1 * likelihood


def calculate_static_ll(params, data):
    l3, d = params[0:2]
    likelihood = 0
    f = np.array(params[2:])
    N = len(f) // 2

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
            f_ij = np.array([f[i] if np.abs(f[i]) < 3 else 3 * np.sign(f[i]) for i in indices])
            l1 = np.exp(d + f_ij[0] - f_ij[3])
            l2 = np.exp(f_ij[1] - f_ij[2])
            x, y = int(match[0]), int(match[1])

            round_ll += np.log(calc_Pbp(x, y, l1, l2, l3))

        likelihood += round_ll

    print(likelihood)
    return -1 * likelihood


def forecast_f(params, init_f, data, PLOT=False):
    a1, a2, b1, b2, l3, d = params
    f = np.copy(init_f)
    N = len(f) // 2
    w = np.multiply(f, np.concatenate((np.repeat(1-b1, N), np.repeat(1-b2, N))))
    A = np.diag([a1, a1, a2, a2])
    B = np.diag([b1, b1, b2, b2])
    output = np.zeros((len(data[:,0]), 9))

    rounds = np.unique(data[:,6])
    ajax = np.zeros(len(rounds))
    fey = np.zeros(len(rounds))
    match_no = 0
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j, i+N, j+N]
            f_ij = np.array([f[i] if np.abs(f[i]) < 3 else 3 * np.sign(f[i]) for i in indices])
            w_ij = np.array([w[i] for i in indices])
            l1 = np.exp(d + f_ij[0] - f_ij[3])
            l2 = np.exp(f_ij[1] - f_ij[2])
            x, y = int(match[0]), int(match[1])

            # Update values
            res = match[2]
            output[match_no, 0] = res
            output[match_no, 1] = f_ij[0]
            output[match_no, 2] = f_ij[1]
            output[match_no, 3] = f_ij[2]
            output[match_no, 4] = f_ij[3]
            output[match_no, 5] = 1
            output[match_no, 6] = 1
            output[match_no, 7] = 1
            output[match_no, 8] = 1

            f_update = update_f_ijt(f_ij, w_ij, A, B, x, y, l1, l2, l3)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

            match_no += 1

        for n in np.where(teams_played == False):
            if f[n][0] != 0: # Do not update teams that have not entered yet
                f[n] = w[n] + b1 * f[n]

        ajax[r - rounds[0]] = f[6]
        fey[r - rounds[0]] = f[14]

    if PLOT:
        sns.lineplot(y=ajax, x=rounds, color='red')
        sns.lineplot(y=fey, x=rounds)
        plt.show()
    return (output, f)

def main():
    # FTHG  FTAG    FTR Season  HomeTeamID  AwayTeamID  RoundNO
    # 0     1       2   3       4           5           6
    data = pd.read_csv("processed_data.csv", usecols=["Season", "RoundNO", "HomeTeamID", "AwayTeamID", "FTR", "FTHG", "FTAG"])
    ftr_mapping = {'A': 0, 'D': 1, 'H': 2}
    data['FTR'] = data['FTR'].apply(lambda x: ftr_mapping[x])
    N = np.max(data['HomeTeamID']) + 1

    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 16
    usable_data = data[data.Season > 0]
    train_data_full = usable_data[usable_data.Season <= split_season]
    train_data = train_data_full.to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)
    

    ESTIMATE_INITIAL_GAMMA = False
    if ESTIMATE_INITIAL_GAMMA:
        bounds = opt.Bounds(np.concatenate((np.array([0, 0]), -3*np.ones(36))), np.concatenate((np.array([3, 3]), 3*np.ones(36))))
        constraint = opt.LinearConstraint(np.concatenate((np.array([0, 0]), np.ones(18), np.zeros(18))), 0, 0)
        k_start = np.array([0.2, 0.5])
        gamma_start = np.repeat(0, 36)
        x0 = np.concatenate((k_start, gamma_start))
        result = opt.minimize(calculate_static_ll, x0, args=(init_data), tol=1e-4, constraints=constraint, bounds=bounds)
        print(result)
        output = pd.DataFrame()
        output["Value"] = result.x
        output.to_csv("initial_f_poisson.csv")
        result = result.x
    else:
        df = pd.read_csv("initial_f_poisson.csv")
        result = np.array(df["Value"])


    ESTIMATE_OVER_ALL_DATA = False
    if ESTIMATE_OVER_ALL_DATA:
        x0 = np.array([0, 0, 0, 0, result[0], result[1]])
        bounds = opt.Bounds(np.array([-0.5, -0.5, -2, -2, 0, 0]), np.array([0.5, 0.5, 2, 2, 1, 1]))
        gamma_init = np.concatenate((np.array(result[2:21]), np.zeros(N - 17), np.array(result[22:-1]), np.zeros(N - 17)))
        all_data_result = opt.minimize(calculate_ll, x0, args=(gamma_init, train_data), tol=1e-6, bounds=bounds)
        print(all_data_result)
        output = pd.DataFrame()
        output["Value"] = all_data_result.x
        output.to_csv("full_poisson_estimates.csv")
        all_data_result = all_data_result.x
    else:
        df = pd.read_csv("full_poisson_estimates.csv")
        all_data_result = np.array(df["Value"])

    FORECAST_ALL_DATA = True
    if FORECAST_ALL_DATA:
        # all_data_result
        output = forecast_f(all_data_result, np.concatenate((np.array(result[2:21]), np.zeros(N - 17), np.array(result[22:-1]), np.zeros(N - 17))), train_data, PLOT=True)[0]
        df_output = pd.DataFrame(output, columns = ["Outcome", "Ha", "Aa", "Hb", "Ab", "ProbA", "ProbD", "ProbH", "Prediction"])
        print(output)
        df_output.to_csv("predictions/poisson_full_16.csv")
        used_data = train_data_full[['FTR', 'Season', 'HomeTeamID', 'AwayTeamID']].to_numpy()
        df_for_NN = pd.DataFrame(np.hstack((used_data, output)), columns = ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', "Outcome", "Ha", "Aa", "Hb", "Ab", "ProbA", "ProbD", "ProbH", "Prediction"])
        df_for_NN.to_csv("poisson_full_NN.csv")

    FORECAST_PER_SEASON = False
    if FORECAST_PER_SEASON:
        x0 = all_data_result
        gamma_init = np.concatenate((np.array(result[2:-1]), np.zeros(N - 17)))
        
        output = np.zeros(1)

        for s in np.unique(test_data[:,3]):
            train_data = usable_data[usable_data.Season <= s - 1].to_numpy(dtype=np.int16)
            parameters = opt.minimize(calculate_ll, x0, args=(gamma_init, train_data), method='Nelder-Mead', tol=1e-2).x
            # USE PARAMETERS
            print(f"Estimation until {s} is complete.")
            _, gamma_init_season = forecast_f(x0, gamma_init, train_data)
            season_forecast, gamma_init = forecast_f(x0, gamma_init_season, test_data[test_data[:,3] == s])

            if s == split_season + 1:
                output = season_forecast
            else:
                output = np.vstack((output, season_forecast))

        df_output = pd.DataFrame(output, columns = ["Outcome", "PowerH", "PowerA", "ProbA", "ProbD", "ProbH", "Prediction"])
        print(output)
        df_output.to_csv("predictions/poisson_per_season.csv")


if __name__ == '__main__':
    main()