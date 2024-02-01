from __future__ import annotations
import pandas as pd
from datetime import datetime
import numpy as np
import dieboldmariano as dm
import glob

def output_type_errors(realizations, forecast):
    errors = np.zeros(9, dtype=np.int16)
    process_count = 0
    for r, f in zip(realizations, forecast):
        if r == 0: # Realized loss
            process_count += 1
            errors[0 + int(f)] = errors[0 + int(f)] + 1
        elif r == 1:# Realized draw
            process_count += 1
            errors[3 + int(f)] = errors[3 + int(f)] + 1
        elif r == 2:# Realized win
            process_count += 1
            errors[6 + int(f)] = errors[6 + int(f)] + 1

    return errors.reshape(3, 3)

def calculate_succes_ratio(realizations, forecast):
    return 1 - np.count_nonzero(realizations - forecast) / len(realizations)


def calculate_RPS(realizations, probabilities):
    rps = np.zeros(len(realizations))
    for i, (o, p) in enumerate(zip(realizations, probabilities)):
        cdf = np.ones(3)
        if int(o) != 2:
            cdf[0] = 0
        if int(o) == 0:
            cdf[1] = 0
        
        squares = np.square([p[0] - cdf[0], p[0] + p[1] - cdf[1], 0])
        rps[i] = np.sum(squares)/2

    return rps


def main():
    ENGLAND = False
    add_string = 'en_' if ENGLAND else ''
    files = glob.glob(f'.\\predictions\\{add_string}df_*.csv')
    results_2 = pd.read_csv(f'predictions\{add_string}poisson_per_season_final.csv')
    pred_2 = results_2['Prediction'].to_numpy()
    
    for file in files:
        results = pd.read_csv(file)
        real = results['Outcome'].to_numpy()
        pred = results['Prediction'].to_numpy()
        prob = results[['ProbA', 'ProbD', 'ProbH']].to_numpy()
        rps = calculate_RPS(real, prob)

        diebold = dm.dm_test(real, pred, pred_2)
        type_errors = output_type_errors(real, pred)
        
        print(file)
        print(f"SR: {calculate_succes_ratio(real, pred):.3f}\tARPS: {np.mean(rps):.3f}\tCH: {int(type_errors[2, 2])}\tCD: {int(type_errors[1, 1])}\tCA: {int(type_errors[0, 0])}\tDMS: {diebold[0]:.3f}\tDMP: {diebold[1]:.3f}")
        print("")

    file = f'predictions\{add_string}probit_per_season_final.csv'
    results = pd.read_csv(f'predictions\{add_string}probit_per_season_final.csv')
    real = results['Outcome'].to_numpy()
    pred = results['Prediction'].to_numpy()
    prob = results[['ProbA', 'ProbD', 'ProbH']].to_numpy()
    rps = calculate_RPS(real, prob)

    diebold = dm.dm_test(real, pred, pred_2)
    type_errors = output_type_errors(real, pred)
    
    print(file)
    print(f"SR: {calculate_succes_ratio(real, pred):.3f}\tARPS: {np.mean(rps):.3f}\tCH: {int(type_errors[2, 2])}\tCD: {int(type_errors[1, 1])}\tCA: {int(type_errors[0, 0])}\tDMS: {diebold[0]:.3f}\tDMP: {diebold[1]:.3f}")
    print("")

    file = f'predictions\{add_string}poisson_per_season_final.csv'
    results = pd.read_csv(f'predictions\{add_string}poisson_per_season_final.csv')
    real = results['Outcome'].to_numpy()
    pred = results['Prediction'].to_numpy()
    prob = results[['ProbA', 'ProbD', 'ProbH']].to_numpy()
    rps = calculate_RPS(real, prob)

    type_errors = output_type_errors(real, pred)
    
    print(file)
    print(f"SR: {calculate_succes_ratio(real, pred):.3f}\tARPS: {np.mean(rps):.3f}\tCH: {int(type_errors[2, 2])}\tCD: {int(type_errors[1, 1])}\tCA: {int(type_errors[0, 0])}")

if __name__ == '__main__':
    main()