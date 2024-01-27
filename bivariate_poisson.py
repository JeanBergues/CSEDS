import numpy as np
import math
import pandas as pd
from scipy.optimize import minimize
from ast import literal_eval

def create_A_B_matrix(a, b, N):
    # Creates first block
    a_block = a * np.eye(N)

    # Create secopnd block
    b_block = b * np.eye(N)

    # Combine the blocks to form matrix
    matrix = np.block([[a_block, np.zeros((N, N))],
                  [np.zeros((N, N)), b_block]])

    return matrix

def pdf_bp(x, y, alpha_it, alpha_jt, beta_it, beta_jt, lambda3, delta):
# Function to calculate the pdf of a bivariate poisson dist
    x = int(x)
    y = int(y)
    # print(delta)
    # print(alpha_it, alpha_jt, beta_it, beta_jt)
    lambda1 = np.exp(delta + alpha_it - beta_jt)
    lambda2 = np.exp(alpha_jt - beta_it)
    product_component = np.exp(-(lambda1+lambda2+lambda3)) * ((lambda1**x)/(math.factorial(x))) * ((lambda2**y)/(math.factorial(y)))
    sum_component = 0
    if min(x,y) == 0:
        sum_component = math.comb(x, 0) * math.comb(y, 0) * math.factorial(0) * (((lambda3)/(lambda1*lambda2))**0)
        return product_component * sum_component
    
    else:
        for k in range(min(x, y)):
            sum_component += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (((lambda3)/(lambda1*lambda2))**k)
            # print(lambda1*lambda2)
            # print(((lambda3)/(lambda1*lambda2))**k)
        # print(sum_component)
        return product_component * sum_component

def S(x, y, lambda1, lambda2, q, lambda3):
    x = int(x)
    y = int(y)
    summation = 0

    if min(x,y) == 0:
        summation = math.comb(x, 0) * math.comb(y, 0) * math.factorial(0) * (0**q) * (((lambda3)/(lambda1*lambda2))**0)

        return summation

    else:
        for k in range(min(x,y)):
            summation += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (k**q) * (((lambda3)/(lambda1*lambda2))**k)
        return summation


def calc_s(x, y, alpha_home, alpha_away, beta_home, beta_away, lambda3, delta):
    # Calc lambda
    lambda1 = np.exp(delta + alpha_home - beta_away)
    lambda2 = np.exp(alpha_away - beta_home)

    # Calc U
    U = (S(x, y, lambda1, lambda2, 1, lambda3) + 1e-8) / (S(x, y, lambda1, lambda2, 0, lambda3) + 1e-8)
    
    # Calc first derivative
    first_der = []
    first_der.append(x - lambda1 - U)
    first_der.append(x - lambda1 - U)
    first_der.append(lambda2 - y + U)
    first_der.append(lambda1 - x + U)

    return first_der

def ll_biv_poisson(params, data, schedule):
# Function for calculating the log likelihood of bivariate poisson dist
    # Define parameters
    a1, a2, b1, b2, lambda3, delta, *f_ini = params

    ####### TO DO
    ####### checkl for NAN for new teams in f_t
    # Length of data
    T = len(data)

    # Setting log likelihood to 0 first
    ll = 0

    # Get list of distinct teams
    all_teams = sorted(schedule['HomeTeam'].unique().tolist())
    nr_teams = len(all_teams)
    
    # Defining f_t, A, B
    f = []
    A = create_A_B_matrix(a1,a2, 2)
    B = create_A_B_matrix(b1,b1, 2)

    # Calculating likelihood
    for t in range(T):
        sum_1 = 0
        if t == 0:
            f.append(f_ini)
            
            # Get all matches from round
            schedule_round = schedule[schedule['round'] == t]

            # Create w
            B_all_teams = create_A_B_matrix(b1,b2, nr_teams)
            w = np.multiply(f[t], (np.ones(len(f[t])) - np.diagonal(B_all_teams)))

            # Create empty list for f_t+1
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round["HomeTeam"][i]
                away = schedule_round["AwayTeam"][i]

                # Get attack strength and defense strength of specific teams from f_t
                # The order of f_t is same as unique list
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Update log likelihood
                sum_1 = np.logaddexp(sum_1, np.log(pdf_bp(data[home][t], data[away][t], f[t][home_index], f[t][away_index], f[t][home_index + nr_teams], f[t][away_index + nr_teams], delta, lambda3))) 
                
                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away

            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season
            
        else:
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            schedule_round = schedule[schedule['round'] == t]
            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round.iloc[i]["HomeTeam"]
                away = schedule_round.iloc[i]["AwayTeam"]

                # Get index of teams
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away
                # Updating sum
                sum_1 = np.logaddexp(sum_1, np.log(pdf_bp(x, y, f[t][home_index], f[t][away_index], f[t][home_index + nr_teams], f[t][away_index + nr_teams], delta, lambda3))) 
                # print(sum_1)
                
            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season
            # print(np.isnan(f[t+1]).any())
        # Updating ll
        ll += sum_1

    print(a1, a2, b1, b2, delta, lambda3, f[int(max(schedule['round']))-1][0], f[int(max(schedule['round']))-1][23])
    print(ll)
    return -ll

def initial_training_model_bp(data, schedule, name_output):
    # initial values a1, a2, b1, b2, lambda3, delta, f_ini, data, schedule
    a1_ini = 0.1
    a2_ini = 0.1
    b1_ini = 0.1
    b2_ini = 0.1
    lambda3_ini = (np.cov(schedule['FTHG'], schedule['FTAG'])[0,0])
    delta_ini = np.log(np.cov(schedule['FTHG'], schedule['FTAG'])[0,0])
    f_ini = [0.3 for i in range(2*len(schedule["HomeTeam"].unique()))]

    # df_ini = pd.read_csv("BP_results_first.csv")
    # a1_ini = df_ini["a1"][0]
    # a2_ini = df_ini["a2"][0]
    # b1_ini = df_ini["b1"][0]
    # b2_ini = df_ini["b2"][0]
    # lambda3_ini = df_ini["lambda3"][0]
    # delta_ini = df_ini["delta"][0]
    # f_ini = literal_eval(df_ini["f"][0])

    initial_values = []
    initial_values.append(a1_ini)
    initial_values.append(a2_ini)
    initial_values.append(b1_ini)
    initial_values.append(b2_ini)
    initial_values.append(lambda3_ini)
    initial_values.append(delta_ini)

    bounds = [(-2,2), (-2,2), (-2,2), (-2,2), (0,10), (-2,2)]
    for i in range(len(f_ini)):
        initial_values.append(f_ini[i])
        bounds.append((-2,2))

    result = minimize(ll_biv_poisson, initial_values, args=(data, schedule,), bounds=bounds, method='Nelder-Mead', options={'maxiter' : 30000})

    print(result)
    est_a1, est_a2, est_b1, est_b2, est_lambda3, est_delta, *est_f = result.x
    df_results = pd.DataFrame({"a1": [est_a1],
                               "a2": [est_a2],
                               "b1": [est_b1],
                               "b2": [est_b2],
                               "lambda3": [est_lambda3],
                               "delta": [est_delta],
                               "f": [est_f]})
    df_results.to_csv(name_output, index=False)

def get_f(data, schedule, params):
    # Define parameters
    a1, a2, b1, b2, lambda3, delta, *f_ini = params

    # Length of data
    T = len(data)

    # Setting log likelihood to 0 first
    ll = 0

    # Get list of distinct teams
    all_teams = sorted(schedule['HomeTeam'].unique().tolist())
    nr_teams = len(all_teams)
    
    # Defining f_t
    f = []

    # Calculating f_t
    for t in range(T):
        sum_1 = 0
        if t == 0:
            f.append(f_ini)
            
            # Get all matches from round
            schedule_round = schedule[schedule['round'] == t]

            # Create w
            B_all_teams = create_A_B_matrix(b1,b2, nr_teams)
            w = np.multiply(f[t], (np.ones(len(f[t])) - np.diagonal(B_all_teams)))

            # Create empty list for f_t+1
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round["HomeTeam"][i]
                away = schedule_round["AwayTeam"][i]

                # Get attack strength and defense strength of specific teams from f_t
                # The order of f_t is same as unique list
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away

            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season
            
        else:
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            schedule_round = schedule[schedule['round'] == t]
            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round.iloc[i]["HomeTeam"]
                away = schedule_round.iloc[i]["AwayTeam"]

                # Get index of teams
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away
                
            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season

    return f

def retrain_bp(data, schedule, ini):
    ini_a1, ini_a2, ini_b1, ini_b2, ini_lambda3, ini_delta, *ini_f = ini

    if 2*len(schedule["HomeTeam"].unique()) != len(ini):
        # get the order of which element belongs to which team in f_t
        order_ini = sorted(schedule[schedule["round"] < max(schedule['round']), "HomeTeam"].unique().tolist())

        # get new order of f_t
        order_new = sorted(schedule["HomeTeam"].unique().tolist())

        f_new = [None for i in range(2*len(schedule["HomeTeam"].unique()))]
        for team in order_new:
            if team in order_ini:
                # get index
                index_ini = order_ini.index(team)
                index_new = order_new.index(team)

                f_new[index_new] = ini_f[index_ini] # Attack
                f_new[index_new + len(order_new)] = ini_f[index_ini + len(order_ini)] # Defense

            else:
                index_new = order_new.index(team)

                f_new[index_new] = np.mean(ini_f[:len(order_ini)]) # Attack
                f_new[index_new + len(order_new)] = np.mean(ini_f[-len(order_ini):]) # Defense


        initial = [ini_a1, ini_a2, ini_b1, ini_b2, ini_lambda3, ini_delta]

        for i in range(len(f_new)):
            initial.append(f_new[i])
        
        bounds = [(-2,2), (-2,2), (-2,2), (-2,2), (0,10), (-2,2)]
        result = minimize(ll_biv_poisson, initial, args=(data, schedule,), bounds=bounds, method='Nelder-Mead', tol=1e-3)
    else:
        bounds = [(-2,2), (-2,2), (-2,2), (-2,2), (0,10), (-2,2)]
        result = minimize(ll_biv_poisson, ini_f, args=(data, schedule,), bounds=bounds, method='Nelder-Mead', tol=1e-3)
    
    return result.x

def calc_probas(home_index, away_index, nr_teams, params):
    a1, a2, b1, b2, lambda3, delta, *f_0 = params

    proba_home_win = 0
    proba_draw = 0
    proba_away_win = 0
    for x in range(26):
        if x > 0:
            for y in range(x):
                # Calc proba home win
                proba_home_win += pdf_bp(x, y, f[home_index], f[away_index], f[home_index + nr_teams], f[away_index + nr_teams], lambda3, delta)

                # Calc proba away win
                proba_away_win += pdf_bp(y, x, f[home_index], f[away_index], f[home_index + nr_teams], f[away_index + nr_teams], lambda3, delta)

        # Calc proba draw
        proba_draw += pdf_bp(x, x, f[home_index], f[away_index], f[home_index + nr_teams], f[away_index + nr_teams], lambda3, delta)
    
    return proba_home_win, proba_draw, proba_away_win

def one_step_ahead_forecast(data, schedule):
    # Get first estimates
    first_results = pd.read_csv("BP_results_second.csv")
    a1 = first_results["a1"]
    a2 = first_results["a2"]
    b1 = first_results["b1"]
    b2 = first_results["b2"]
    lambda3 = first_results["lambda3"][0]
    delta = first_results["delta"][0]
    f = literal_eval(first_results["f"][0])

    params = [a1, a2, b1, b2, lambda3, delta]
    for i in range(len(f)):
        params.append(f[i])

    # Creating dataframe with results
    proba_df = pd.DataFrame(np.nan, index=range(10000), columns=['HomeTeam', 'AwayTeam', 'Proba_Home_win', 'Proba_Draw', 'Proba_Away_win', 'round'])

    count = 0
    # Get matches that need to be forecasted
    for i in range(173, max(schedule["round"] + 1)):
        round_matches = schedule[schedule['round'] == i]
        train_schedule = schedule.head(i)
        teams = sorted(train_schedule["HomeTeam"].unique().tolist())

        for k in range(len(round_matches)):
            home = round_matches.loc[k, "HomeTeam"]
            away = round_matches.loc[k, "AwayTeam"]

            home_index = teams.index(home)
            away_index = teams.index(away)

            f = get_f(test_data, test_schedule, params)

            proba_home_win, proba_draw, proba_away_win = calc_probas(home_index, away_index, len(teams), params)

            # Update dataframe
            proba_df.loc[count, "HomeTeam"] = home
            proba_df.loc[count, "AwayTeam"] = away
            proba_df.loc[count, "Proba_Home_win"] = proba_home_win
            proba_df.loc[count, "Proba_Draw"] = proba_draw
            proba_df.loc[count, "Proba_Away_win"] = proba_away_win
            proba_df.loc[count, "round"] = home
            count += 1

        # Retraining model
        params = retrain_bp(data.head(i), train_schedule, params)
    
    proba_df.to_csv("One_step_ahead_forecasts.csv", index=False)
    return 


# Read Data
schedule = pd.read_csv("BP_data_NEW/schedule.csv")
data = pd.read_csv("BP_data_NEW/panel_data.csv")

# Train model
# Training model on whole data set for ANN
# initial_training_model_bp(data, schedule, "BP_results_for_NN.csv")

# Training model on first training set
# initial_training_model_bp(data.head(752), schedule[schedule["round"] < 752], "BP_results_second.csv")

# last round first 20 seasons is 751
# Solution for new team in test data
# Get training data unique teams
# check if home , away is in this unique team
# if not get set alpha of this team to average attack and defense
