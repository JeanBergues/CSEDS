import numpy as np
import math
import pandas as pd
from scipy.optimize import minimize

def constraint1(f):
    return sum(f[len/2:])

def constraint2(f):
    return sum(f[:len/2])
               
def pdf_x_ini_value(x, alpha_it, beta_jt):
    return -alpha_it*beta_jt + x*np.log(alpha_it*beta_jt) - np.log(math.factorial(x))

def pdf_y_ini_value(y, alpha_jt, beta_it):
    return -alpha_jt*beta_it + y*np.log(alpha_jt*beta_it) - np.log(math.factorial(y))

def ll_x_ini_value(f_ini, data):
    # Get list of distinct teams
    all_teams = data["HomeTeam"].unique().tolist()

    sum_1 = 0
    for i in range(len(data)):
        # Get match opponents
        home = data["HomeTeam"][i]
        away = data["AwayTeam"][i]

        # Get attack strength and defense strength of specific teams from f_t
        # The order of f_t is same as unique list
        home_index = all_teams.index(home)
        away_index = all_teams.index(away)

        sum_1 += pdf_x_ini_value(data["FTHG"][i], f_ini[home_index], f_ini[away_index + len(all_teams)])

    return -sum_1

def ll_y_ini_value(f_ini, data):
    # Get list of distinct teams
    all_teams = data["HomeTeam"].unique().tolist()

    sum_1 = 0
    for i in range(len(data)):
        # Get match opponents
        home = data["HomeTeam"][i]
        away = data["AwayTeam"][i]

        # Get attack strength and defense strength of specific teams from f_t
        # The order of f_t is same as unique list
        home_index = all_teams.index(home)
        away_index = all_teams.index(away)

        sum_1 += pdf_x_ini_value(data["FTAG"][i], f_ini[away_index], f_ini[home_index + len(all_teams)])
    return -sum_1

df = pd.read_csv(r'C:\Users\Romni\Documents\GitHub\CSEDS\data_nl\season0001.csv')

all_teams = df["HomeTeam"].unique().tolist()

sum_constraint = [{'type': 'eq', 'fun': constraint1},
                  {'type': 'eq', 'fun': constraint2}]

S = df['FTHG'].sum
for i in range(len(all_teams)):
    

f_ini = [0.1 for i in range(2*len(all_teams))]
result_x = minimize(ll_x_ini_value, f_ini, args=(df,), method='BFGS', constraints=sum_constraint)
result_y = minimize(ll_y_ini_value, f_ini, args=(df,), method='BFGS', constraints=sum_constraint)

print(result_x)
print(result_y)


