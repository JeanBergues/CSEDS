import pandas as pd
import numpy as np
import scipy.optimize as opt
import scipy.stats as sts
import warnings
warnings.filterwarnings("error")

# DEFINITIONS
# f = [a1, a2, ..., b1, b2]
# phi = [a1, a2, b1, b2, l3, d]

### Mathematical functions
def calc_Pbp(C, l6, k1, k2):
    if C == 2:
        return sts.norm.cdf(k1 - l6) + 1e-12
    if C == 1:
        return sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6) + 1e-12
    if C == 0:
        return 1 - sts.norm.cdf(k2 - l6) + 1e-12


def calc_Delta_ijt(C, l6, k1, k2):
    if C == 2:
        elem_1 = sts.norm.pdf(k2 - l6) / (1 - sts.norm.cdf(k2 - l6) + 1e-12)
        elem_2 = -1*sts.norm.pdf(k2 - l6) / (1 - sts.norm.cdf(k2 - l6) + 1e-12)
        return np.array([elem_1, elem_2])
    if C == 1:
        elem_1 = (sts.norm.pdf(k1 - l6) - sts.norm.pdf(k2 - l6)) / (sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6) + 1e-12)
        elem_2 = (sts.norm.pdf(k2 - l6) - sts.norm.pdf(k1 - l6)) / (sts.norm.cdf(k2 - l6) - sts.norm.cdf(k1 - l6) + 1e-12)
        return np.array([elem_1, elem_2])
    if C == 0:
        elem_1 = -1*sts.norm.pdf(k1 - l6) / (1 - sts.norm.cdf(k1 - l6) + 1e-12)
        elem_2 = sts.norm.pdf(k1 - l6) / (1 - sts.norm.cdf(k1 - l6) + 1e-12)
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

        for n in np.where(teams_played == False):
            if f[n][0] != 0: # Do not update teams that have not entered yet
                f[n] = w[n] + b1 * f[n]

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

        likelihood += round_ll

    print(likelihood)
    return -1 * likelihood


def forecast_f(params, init_f, data):
    a1, b1, k1, k2star = params
    k2 = k1 + np.exp(k2star)
    f = np.copy(init_f)
    N = len(f)
    w = np.multiply(f, np.repeat(1-b1, N))
    output = np.zeros((len(data[:,0]), 7))
    match_no = 0
    A = np.diag([a1, a1])
    B = np.diag([b1, b1])

    rounds = np.unique(data[:,6])
    for r in rounds:
        teams_played = np.repeat(False, N)
        r_data = data[data[:, 6] == r]

        for match in r_data:
            i, j = match[4], match[5]
            teams_played[i] = True
            teams_played[j] = True

            indices = [i, j]
            f_ij = np.array([f[i] for i in indices])
            w_ij = np.array([w[i] for i in indices])
            
            C = match[2]
            l6 = f_ij[0] - f_ij[1]

            # Update values
            output[match_no, 0] = C
            output[match_no, 1] = f_ij[0]
            output[match_no, 2] = f_ij[1]
            output[match_no, 3] = calc_Pbp(0, l6, k1, k2)
            output[match_no, 4] = calc_Pbp(1, l6, k1, k2)
            output[match_no, 5] = calc_Pbp(2, l6, k1, k2)
            output[match_no, 6] = np.argmax([calc_Pbp(x, l6, k1, k2) for x in [0, 1, 2]])

            f_update = update_f_ijt(f_ij, w_ij, A, B, C, l6, k1, k2)

            for x, index in enumerate(indices):
                f[index] = f_update[x]

            match_no += 1

        for n in np.where(teams_played == False):
            if f[n][0] != 0: # Do not update teams that have not entered yet
                f[n] = w[n] + b1 * f[n]
            
    return (output, f)

def main():
    # FTHG  FTAG    FTR Season  HomeTeamID  AwayTeamID  RoundNO
    # 0     1       2   3       4           5           6
    data = pd.read_csv("processed_data.csv", usecols=["Season", "RoundNO", "HomeTeamID", "AwayTeamID", "FTR", "FTHG", "FTAG"])
    ftr_mapping = {'A': 0, 'D': 1, 'H': 2}
    data['FTR'] = data['FTR'].apply(lambda x: ftr_mapping[x])
    N = np.max(data['HomeTeamID']) + 1

    init_data = data[data.Season == 0].to_numpy(dtype=np.int16)
    
    split_season = 18
    usable_data = data[data.Season > 0]
    train_data_full = usable_data[usable_data.Season <= split_season]
    train_data = train_data_full.to_numpy(dtype=np.int16)
    test_data = usable_data[usable_data.Season > split_season].to_numpy(dtype=np.int16)
    

    ESTIMATE_INITIAL_GAMMA = True
    if ESTIMATE_INITIAL_GAMMA:
        constraint = opt.LinearConstraint(np.concatenate((np.array([0, 0]), np.ones(18))).T, 0, 0)
        k_start = np.array([1, 1])
        gamma_start = np.repeat(0, 18)
        x0 = np.concatenate((k_start, gamma_start))
        result = opt.minimize(calculate_static_ll, x0, args=(init_data), tol=1e-4, constraints=constraint)
        print(result)
        output = pd.DataFrame()
        output["Value"] = result.x
        output.to_csv("initial_gamma_probit_final.csv")
        result = result.x
    else:
        df = pd.read_csv("initial_gamma_probit_final.csv")
        result = np.array(df["Value"])


    ESTIMATE_OVER_ALL_DATA = True
    if ESTIMATE_OVER_ALL_DATA:
        x0 = np.array([0.5, 0.5, result[0], result[1]])
        gamma_init = np.concatenate((np.array(result[2:-1]), np.zeros(N - 17)))
        all_data_result = opt.minimize(calculate_ll, x0, args=(gamma_init, train_data), tol=1e-2)
        print(all_data_result)
        output = pd.DataFrame()
        output["Value"] = all_data_result.x
        output.to_csv("full_probit_estimates_final.csv")
        all_data_result = all_data_result.x
    else:
        df = pd.read_csv("full_probit_estimates_final.csv")
        all_data_result = np.array(df["Value"])

    FORECAST_ALL_DATA = True
    if FORECAST_ALL_DATA:
        output = forecast_f(all_data_result, np.concatenate((np.array(result[2:-1]), np.zeros(N - 17))), train_data)[0]
        df_output = pd.DataFrame(output, columns = ["Outcome", "PowerH", "PowerA", "ProbA", "ProbD", "ProbH", "Prediction"])
        print(output)
        df_output.to_csv("predictions/probit_full_final.csv")
        used_data = train_data_full[['FTR', 'Season', 'HomeTeamID', 'AwayTeamID']].to_numpy()
        df_for_NN = pd.DataFrame(np.hstack((used_data, output)), columns = ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', "Outcome", "PowerH", "PowerA", "ProbA", "ProbD", "ProbH", "Prediction"])
        df_for_NN.to_csv("probit_full_NN_final.csv")

    FORECAST_PER_SEASON = True
    if FORECAST_PER_SEASON:
        x0 = all_data_result
        gamma_init = np.concatenate((np.array(result[2:-1]), np.zeros(N - 17)))
        
        output = np.zeros(1)

        for s in np.unique(test_data[:,3]):
            train_data = usable_data[usable_data.Season <= s - 1].to_numpy(dtype=np.int16)
            parameters = opt.minimize(calculate_ll, x0, args=(gamma_init, train_data), method='Nelder-Mead', tol=1e-2).x
            # USE PARAMETERS
            print(f"Estimation until {s} is complete.")
            _, gamma_init_season = forecast_f(parameters, gamma_init, train_data)
            season_forecast, gamma_init = forecast_f(parameters, gamma_init_season, test_data[test_data[:,3] == s])

            if s == split_season + 1:
                output = season_forecast
            else:
                output = np.vstack((output, season_forecast))

        df_output = pd.DataFrame(output, columns = ["Outcome", "PowerH", "PowerA", "ProbA", "ProbD", "ProbH", "Prediction"])
        print(output)
        df_output.to_csv("predictions/probit_per_season_final.csv")


if __name__ == '__main__':
    main()