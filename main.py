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
import datetime
import sklearn.neural_network as nn


def read_single_csv(file_name: str, columns: list[str]) -> pd.DataFrame:
    # print(f"Now reading file {file_name}")
    return pd.read_csv(file_name, sep=',', usecols=columns)


def read_and_filter_data(country: str, columns: list[str]) -> pd.DataFrame:
    files = glob.glob(f'.\\data_{country}\\*.csv')
    
    dataframes = [read_single_csv(file, columns) for file in files]
    return pd.concat(dataframes)


def train_neural_network(data: pd.DataFrame, y_name: str, no_hidden_layers: int, hidden_layer_size: int):
    # TODO: tweak other parameters in regressor function
    neural_network = nn.MLPRegressor(hidden_layer_sizes=(no_hidden_layers, hidden_layer_size))
    neural_network.fit(data.drop(y_name, axis=1), data[y_name])


def main() -> None:
    # Set-up variables concerning data
    country = 'nl'
    selected_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    # Set-up variables concerning neural network
    no_hidden_layers = 1
    hidden_layer_size = 10
    
    # Read in the data
    data = read_and_filter_data(country, selected_columns)
    # print(data)

    # Prepare the data for training a neural network
    # Make dates an ordinal integer value
    #data["Date"].apply(lambda d: datetime.date.toordinal(d))

    # Factorize team names
    # _, mapping = pd.
    data[["HomeTeam", "AwayTeam"]] = data[["HomeTeam", "AwayTeam"]].apply(lambda x: pd.factorize(x)[0])
    print(data)

    # Factorize scores

    # Train the neural network
    # train_neural_network(data.drop('Date', axis=1), 'FTR', no_hidden_layers, hidden_layer_size)


if __name__ == "__main__":
    main()