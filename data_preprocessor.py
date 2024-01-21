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
from datetime import datetime
import sklearn.neural_network as nn


def read_single_csv(file_name: str, columns: list[str]) -> pd.DataFrame:
    # print(f"Now reading file {file_name}")
    df = pd.read_csv(file_name, sep=',', usecols=columns)
    df = df.dropna(axis=0)

    # Already convert date to ordinal values, as after 2018 the time format changes
    date_format = '%d/%m/%Y' if int(file_name[16:20]) >= 1819 else '%d/%m/%y'
    df["ORDDate"] = df["Date"].apply(lambda d: datetime.strptime(d, date_format).date().toordinal())
    return df


def read_and_filter_data(country: str, columns: list[str]) -> pd.DataFrame:
    files = glob.glob(f'.\\data_{country}\\*.csv')
    
    dataframes = [read_single_csv(file, columns) for file in files]
    return pd.concat(dataframes)


def train_neural_network_classifier(data: pd.DataFrame, y_name: str, no_hidden_layers: int, hidden_layer_size: int):
    # TODO: tweak other parameters in regressor function
    neural_network = nn.MLPClassifier(hidden_layer_sizes=(no_hidden_layers, hidden_layer_size))
    neural_network.fit(data.drop(y_name, axis=1), data[y_name])
    return neural_network


def main() -> None:
    # Set-up variables concerning data
    country = 'nl'
    selected_columns = ["Date", "HomeTeam", "AwayTeam", "FTR"]

    # Set-up variables concerning neural network
    no_hidden_layers = 2
    hidden_layer_size = 20
    
    # Read in the data
    data = read_and_filter_data(country, selected_columns)
    
    # Prepare the data for training a neural network
    # Strip leading and trailing whitespace out of club names to prevent duplicates
    data[["HomeTeam", "AwayTeam"]] = data[["HomeTeam", "AwayTeam"]].apply(lambda x: x.str.strip())

    # Normalize ordinal dates to cut down on numerical issues
    starting_ordinal_date = data['ORDDate'].min()
    data['ORDDate'] = data['ORDDate'].apply(lambda x: x - starting_ordinal_date)

    # Factorize HomeTeam
    data['HomeTeam'], team_mapping = pd.factorize(data['HomeTeam'])
    data['AwayTeam'] = data['AwayTeam'].apply(lambda x: team_mapping.get_loc(x))

    # Train the neural network
    class_nn = train_neural_network_classifier(data.drop('Date', axis=1), 'FTR', no_hidden_layers, hidden_layer_size)

    # Predict one specific game
    print(team_mapping)
    pred_hometeam = team_mapping.get_loc('Ajax')
    pred_awayteam = team_mapping.get_loc('Nijmegen')
    pred_date = '22/01/2024'

    input_vec = pd.DataFrame(
        {
            'HomeTeam': pred_hometeam, 
            'AwayTeam': pred_awayteam,
            'ORDDate': datetime.strptime(pred_date, '%d/%m/%Y').date().toordinal(),
        }, index=[0])

    print(class_nn.predict(input_vec))
    print(data)

if __name__ == "__main__":
    main()