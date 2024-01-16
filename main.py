# Case study in Econometrics and Data science
# Comparing football match forecasting performance between neural networks and score-driven time series models
# Group 3
# Date: 16/01/2024

# Authors:
# Jan Koolbergen    - 2667333

from __future__ import annotations
import glob

import pandas as pd
import numpy as np


def read_single_csv(file_name: str, columns: list[str]) -> pd.DataFrame:
    print(f"Now reading file {file_name}")
    return pd.read_csv(file_name, sep=',', usecols=columns)


def read_and_filter_data(country: str, columns: list[str]) -> pd.DataFrame:
    files = glob.glob(f'.\\data_{country}\\*.csv')
    
    dataframes = [read_single_csv(file, columns) for file in files]
    return pd.concat(dataframes)


def main() -> None:
    # Set-up variables
    country = 'nl'
    selected_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    
    # Read in the data
    data = read_and_filter_data(country, selected_columns)
    print(data)


if __name__ == "__main__":
    main()