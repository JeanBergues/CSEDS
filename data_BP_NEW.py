import numpy as np
import math
import pandas as pd

def all_data():
    # Reading all data
    data = pd.read_csv("processed_data.csv")

    data = data.dropna(subset=["HomeTeam", "AwayTeam"])
    data.reset_index()
    data["round"] = np.nan
    count = 0
    occured_teams = []
    for i in range(len(data)):
        if data.iloc[i]["HomeTeam"] in occured_teams or data.iloc[i]["AwayTeam"] in occured_teams:
            count += 1
            occured_teams = []
            occured_teams.append(data.iloc[i]["HomeTeam"])
            occured_teams.append(data.iloc[i]["AwayTeam"])
        else:
            occured_teams.append(data.iloc[i]["HomeTeam"])
            occured_teams.append(data.iloc[i]["AwayTeam"])
        data.loc[i, "round"] = count
    
    # Cleaning when the same club is written differently. Now all teams that occur in HomeTeam are the same as AwayTeam
    data.loc[data["AwayTeam"] == "Roda JC","AwayTeam"] = "Roda"
    data.loc[data["AwayTeam"] == "Sparta Rotterdam","AwayTeam"] = "Sparta"

    data.loc[data["HomeTeam"] == "Roda JC","HomeTeam"] = "Roda"
    data.loc[data["HomeTeam"] == "Roda ","HomeTeam"] = "Roda"
    data.loc[data["HomeTeam"] == "Ajax ","HomeTeam"] = "Ajax"
    data.loc[data["HomeTeam"] == "Graafschap ","HomeTeam"] = "Graafschap"
    data.loc[data["HomeTeam"] == "Groningen ","HomeTeam"] = "Groningen"
    data.loc[data["HomeTeam"] == "Utrecht ","HomeTeam"] = "Utrecht"
    data.loc[data["HomeTeam"] == "Vitesse ","HomeTeam"] = "Vitesse"
    data.loc[data["HomeTeam"] == "Willem II ","HomeTeam"] = "Willem II"
    data.loc[data["HomeTeam"] == "Sparta Rotterdam","HomeTeam"] = "Sparta"
    data.loc[data["HomeTeam"] == "Heracles ","HomeTeam"] = "Heracles"
    data.loc[data["HomeTeam"] == "Feyenoord ","HomeTeam"] = "Feyenoord"

    data.reset_index()
    data.to_csv("BP_data_NEW\schedule.csv", index=False)

    return

def create_panel_data():
    df = pd.read_csv("BP_data_NEW\schedule.csv")

    # Get distinct teams
    teams = df['HomeTeam'].unique()
    
    # Initialize an empty DataFrame with teams as columns
    result_df = pd.DataFrame(columns=teams, index=range(int(max(df["round"]) +1)))

    # Iterate over rounds and teams to fill in the goals
    for round_num in sorted(df['round'].unique()):
        round_data = df[df['round'] == round_num]

        for i in range(len(round_data)):
            # print(round_data["HomeTeam"][i])
            
            home = round_data.iloc[i]["HomeTeam"]
            away = round_data.iloc[i]["AwayTeam"]
   
            result_df.loc[round_num, home] = round_data.iloc[i]["FTHG"]
            result_df.loc[round_num, away] = round_data.iloc[i]["FTAG"]
    
    result_df.to_csv("BP_data_NEW\panel_data.csv")

all_data()
create_panel_data()

