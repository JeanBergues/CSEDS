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
import numpy as np


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


def transform_result_into_number(result, htID, atID, clubID):
    if result == 'A':
        if atID == clubID:
            return 1
        else:
            return -1
        
    elif result == 'H':
        if htID == clubID:
            return 1
        else:
            return -1
        
    else:
        return 0


def find_previous_club_result(data: pd.DataFrame, clubID, game_ord_date):
    older_data = data[data['DateNR'] < game_ord_date]
    try:
        last_game = older_data[(older_data['HomeTeamID'] == clubID) | (older_data['AwayTeamID'] == clubID)].iloc[-1]
    except IndexError as e:
        return 0
    return transform_result_into_number(last_game['FTR'], last_game['HomeTeamID'], last_game['AwayTeamID'], clubID)


def find_previous_duel_result(data: pd.DataFrame, htID, atID, game_ord_date):
    older_data = data[data['DateNR'] < game_ord_date]
    try:
        last_game = older_data[((older_data['HomeTeamID'] == htID) & (older_data['AwayTeamID'] == atID)) | ((older_data['HomeTeamID'] == atID) & (older_data['AwayTeamID'] == htID))].iloc[-1]
    except IndexError as e:
        return 0
    if last_game['HomeTeamID'] == htID:
        return transform_result_into_number(last_game['FTR'], htID, atID, htID)
    else:
        return transform_result_into_number(last_game['FTR'], atID, htID, htID)


def main() -> None:
    # Set-up variables concerning data
    country = 'nl'
    selected_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    
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

    # Add last match results
    N = data["DateNR"].size
    prev_ht_results = np.zeros(N)
    prev_at_results = np.zeros(N)
    prev_duel_results = np.zeros(N)

    for i, row in data.iterrows():
        prev_ht_results[i] = find_previous_club_result(data, row['HomeTeamID'], row["DateNR"])
        prev_at_results[i] = find_previous_club_result(data, row['AwayTeamID'], row["DateNR"])
        prev_duel_results[i] = find_previous_duel_result(data, row['HomeTeamID'], row['AwayTeamID'], row["DateNR"])

    data['PrevHTR'] = prev_ht_results
    data['PrevATR'] = prev_at_results
    data['PrevDR'] = prev_duel_results

    # Export the data
    print(data.columns)
    print(starting_ordinal_date)
    print(team_mapping)
    data.to_csv('processed_data.csv')

if __name__ == "__main__":
    main()