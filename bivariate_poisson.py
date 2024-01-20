import numpy as np
import math
import pandas as pd

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
    lambda1 = np.exp(delta + alpha_it - beta_jt)
    lambda2 = np.exp(alpha_jt - beta_it)
    product_component = np.exp(-(lambda1+lambda2+lambda3)) * ((lambda1**x)/(math.factorial(x))) * ((lambda2**y)/(math.factorial(y)))

    sum_component = 0
    for k in range(min(x, y)):
        sum_component += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * (((lambda3)/(lambda1*lambda2))**k)

    return product_component * sum_component

def calc_s():
    
    return

def ll_biv_poisson(data, schedule, a1, a2, b1, b2, lambda3, delta):
# Function for calculating the log likelihood of bivariate poisson dist
    # Length of data
    N = len(X[0])
    T = len(X)

    # Setting log likelihood to 0 first
    ll = 0

    # Get list of distinct teams
    all_teams = data['Teams'].to_list()
    
    # Defining f_t, A, B
    f = []
    A = create_A_B_matrix(a1,a2, 2)
    B = create_A_B_matrix(b1,b1, 2)

    # TO DO
    # Define s_t

    # Calculating likelihood
    for t in range(T):
        sum_1 = 0
        if t == 0:
            f.append(calc_ini_f)

            schedule_round = schedule[schedule['round'] == t]
            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round[i]["HomeTeam"]
                away = schedule_round[i]["AwayTeam"]

                # Get attack strength and defense strength of specific teams from f_t
                # The order of f_t is same as unique list
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                sum_1 += np.log(pdf_bp(data[t][home], data[t][away], f[t][home_index], f[t][away_index], f[t][home_index + len(f[0])], f[t][away_index + len(f[0])], delta, lambda3))

            w = np.multiply(f[t], (np.ones(len(f[t])) - np.diagonal(B)))

        else:
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append[empty_list]

            schedule_round = schedule[schedule['round'] == t]
            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round[i]["HomeTeam"]
                away = schedule_round[i]["AwayTeam"]

                # Get index of teams
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Update f_t
                f[t][home_index] = w[home_index] + B * f[t-1][home_index] + A * s # Attack strength home
                f[t][away_index] = w[away_index] + B * f[t-1][away_index] + A * s # Attack strength away
                f[t][home_index + len(f[0])] = w[home_index + len(f[0])] + B * f[t-1][home_index + len(f[0])] + A * s # Defense strength home
                f[t][away_index + len(f[0])] = w[away_index + len(f[0])] + B * f[t-1][away_index + len(f[0])] + A * s # Defense strength away

                # Updating sum
                sum_1 += np.log(pdf_bp(data[home][t], data[away][t], f[t][home_index], f[t][away_index], f[t][home_index + len(f[0])], f[t][away_index + len(f[0])], delta, lambda3))

            
            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t][i] = w[i] + b1 * f[t-1][i]
                f[t][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t-1][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season
        
        # Updating ll
        ll += sum_1

    return ll

def calc_ini_f():
    return

def train_model_bp(X, Y):
    f = []
    f.append(calc_ini_f())


data = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
print((data["HomeTeam"].to_list() + data["AwayTeam"].to_list())[307])
print(data.iloc[1]["AwayTeam"])
print(len(data["HomeTeam"]))

