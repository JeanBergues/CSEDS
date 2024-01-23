import numpy as np
import math
import pandas as pd

def all_data():
    # Reading all data
    data1 = pd.read_csv("data_nl\season0000.csv",on_bad_lines='skip')
    data2 = pd.read_csv("data_nl\season0001.csv",on_bad_lines='skip')
    data3 = pd.read_csv("data_nl\season0102.csv",on_bad_lines='skip')
    data4 = pd.read_csv("data_nl\season0203.csv",on_bad_lines='skip')
    data5 = pd.read_csv("data_nl\season0304.csv",on_bad_lines='skip')
    data6 = pd.read_csv("data_nl\season0405.csv",on_bad_lines='skip')
    data7 = pd.read_csv("data_nl\season0506.csv",on_bad_lines='skip')
    data8 = pd.read_csv("data_nl\season0607.csv",on_bad_lines='skip')
    data9 = pd.read_csv("data_nl\season0708.csv",on_bad_lines='skip')
    data10 = pd.read_csv("data_nl\season0809.csv",on_bad_lines='skip')
    data11 = pd.read_csv("data_nl\season0910.csv",on_bad_lines='skip')
    data12 = pd.read_csv("data_nl\season1011.csv",on_bad_lines='skip')
    data13 = pd.read_csv("data_nl\season1112.csv",on_bad_lines='skip')
    data14 = pd.read_csv("data_nl\season1213.csv",on_bad_lines='skip')
    data15 = pd.read_csv("data_nl\season1314.csv",on_bad_lines='skip')
    data16 = pd.read_csv("data_nl\season1415.csv",on_bad_lines='skip')
    data17 = pd.read_csv("data_nl\season1516.csv",on_bad_lines='skip')
    data18 = pd.read_csv("data_nl\season1617.csv",on_bad_lines='skip')
    data19 = pd.read_csv("data_nl\season1718.csv",on_bad_lines='skip')
    data20 = pd.read_csv("data_nl\season1819.csv",on_bad_lines='skip')
    data21 = pd.read_csv("data_nl\season1920.csv",on_bad_lines='skip')
    data22 = pd.read_csv("data_nl\season2021.csv",on_bad_lines='skip')
    data23 = pd.read_csv("data_nl\season2122.csv",on_bad_lines='skip')
    data24 = pd.read_csv("data_nl\season2223.csv",on_bad_lines='skip')
    data25 = pd.read_csv("data_nl\season2324.csv",on_bad_lines='skip')

    dfs = [data1, data2, data3, data4, data5,
           data6, data7, data8, data9, data10,
           data11, data12, data13, data14, data15,
           data16, data17, data18, data19, data20,
           data21, data22, data23, data24, data25]
    count = 1
    for i in dfs:
        i["Dataset"] = 1
        count += 1
    
    data = pd.concat(dfs, ignore_index=True)
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
    
    data.to_csv("All_data_with_rounds.csv", index=False)
    return

def clean_all_data():
    df = pd.read_csv("All_data_with_rounds.csv")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    
    # Cleaning when the same club is written differently. Now all teams that occur in HomeTeam are the same as AwayTeam
    df.loc[df["AwayTeam"] == "Roda JC","AwayTeam"] = "Roda"
    df.loc[df["AwayTeam"] == "Sparta Rotterdam","AwayTeam"] = "Sparta"

    df.loc[df["HomeTeam"] == "Roda JC","HomeTeam"] = "Roda"
    df.loc[df["HomeTeam"] == "Roda ","HomeTeam"] = "Roda"
    df.loc[df["HomeTeam"] == "Ajax ","HomeTeam"] = "Ajax"
    df.loc[df["HomeTeam"] == "Graafschap ","HomeTeam"] = "Graafschap"
    df.loc[df["HomeTeam"] == "Groningen ","HomeTeam"] = "Groningen"
    df.loc[df["HomeTeam"] == "Utrecht ","HomeTeam"] = "Utrecht"
    df.loc[df["HomeTeam"] == "Vitesse ","HomeTeam"] = "Vitesse"
    df.loc[df["HomeTeam"] == "Willem II ","HomeTeam"] = "Willem II"
    df.loc[df["HomeTeam"] == "Sparta Rotterdam","HomeTeam"] = "Sparta"
    df.loc[df["HomeTeam"] == "Heracles ","HomeTeam"] = "Heracles"
    df.loc[df["HomeTeam"] == "Feyenoord ","HomeTeam"] = "Feyenoord"

    df.reset_index()
    df.to_csv("clean_all_data_with_rounds.csv", index=False)

def create_panel_data():
    df = pd.read_csv("clean_all_data_with_rounds.csv")

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
        
        # for team in teams:
        #     # Checking if team plays in round. If team plays store it in new dataframe
        #     if not round_data[round_data["HomeTeam"] == team].empty:
        #         result_df.at[round_num, team] = round_data.loc[round_data["HomeTeam"] == team, 'FTHG']
                
        #     elif not round_data[round_data["AwayTeam"] == team].empty:
        #         result_df.at[round_num, team] = round_data.loc[round_data["AwayTeam"] == team,'FTAG']
                
        #     else:
        #         result_df.at[round_num, team] = np.nan
    
    result_df.to_csv("panel_data.csv")

