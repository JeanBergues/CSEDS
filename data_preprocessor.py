# Case study in Econometrics and Data science
# Comparing football match forecasting performance between neural networks and score-driven time series models
# Group 3
# Date: 16/01/2024

# Authors:
# Jan Koolbergen    - 2667333

from __future__ import annotations
import glob
import pandas as pd
from datetime import datetime


def read_single_csv(file_name: str, columns: list[str]) -> pd.DataFrame:
    # print(f"Now reading file {file_name}")
    df = pd.read_csv(file_name, sep=',', usecols=columns)
    df = df.dropna(axis=0)

    # Already convert date to ordinal values, as after 2018 the time format changes
    date_format = '%d/%m/%Y' if int(file_name[16:20]) >= 1819 else '%d/%m/%y'
    df["DateNR"] = df["Date"].apply(lambda d: datetime.strptime(d, date_format).date().toordinal())
    return df


def read_and_filter_data(country: str, columns: list[str]) -> pd.DataFrame:
    files = glob.glob(f'.\\data_{country}\\*.csv')
    
    dataframes = [read_single_csv(file, columns) for file in files]
    return pd.concat(dataframes)


def main() -> None:
    # Set-up variables concerning data
    country = 'nl'
    selected_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    # Set-up variables concerning neural network
    no_hidden_layers = 2
    hidden_layer_size = 20
    
    # Read in the data
    data = read_and_filter_data(country, selected_columns)
    
    # Strip leading and trailing whitespace out of club names to prevent duplicates
    data[["HomeTeam", "AwayTeam"]] = data[["HomeTeam", "AwayTeam"]].apply(lambda x: x.str.strip())

    # Normalize ordinal dates to cut down on numerical issues
    starting_ordinal_date = data["DateNR"].min()
    data["DateNR"] = data["DateNR"].apply(lambda x: x - starting_ordinal_date)

    # Factorize HomeTeam and apply the same mapping to AwayTeam
    data['HomeTeamID'], team_mapping = pd.factorize(data['HomeTeam'])
    data['AwayTeamID'] = data['AwayTeam'].apply(lambda x: team_mapping.get_loc(x))

    # Add a score-difference column
    data['ScoreDiff'] = data['FTHG'] - data['FTAG']

    # Export the data
    print(data.columns)
    print(starting_ordinal_date)
    print(team_mapping)
    data.to_csv('processed_data.csv')

if __name__ == "__main__":
    main()