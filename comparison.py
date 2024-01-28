from __future__ import annotations
import glob
import pandas as pd
from datetime import datetime
import numpy as np
import dieboldmariano as dm

def output_type_errors(realizations, forecast):
    errors = np.zeros(9, dtype=np.int16)
    process_count = 0
    for r, f in zip(realizations, forecast):
        if r == 0: # Realized loss
            process_count += 1
            errors[0 + f] = errors[0 + f] + 1
        elif r == 1:# Realized draw
            process_count += 1
            errors[3 + f] = errors[3 + f] + 1
        elif r == 2:# Realized win
            process_count += 1
            errors[6 + f] = errors[6 + f] + 1

    return errors.reshape(3, 3)


def main():
    # Forecast 1
    results = pd.read_csv('predictions/df_nn_class_10_0_.csv')
    real = results['Outcome'].to_numpy()
    pred = results['Prediction'].to_numpy()
    prob = results[['ProbA', 'ProbD', 'ProbH']].to_numpy()

    print(output_type_errors(real, pred))
    


if __name__ == '__main__':
    main()