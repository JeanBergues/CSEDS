import numpy as np 
import pandas as pd

df = pd.read_csv("BP_One_season_ahead_TEST.csv")
df = df.reset_index()
count = 0
for i in range(len(df)):
    proba_home = df.loc[i, "Proba_Home_win"]
    proba_draw = df.loc[i, "Proba_Draw"]
    proba_away = df.loc[i, "Proba_Away_win"]

    if df.loc[i, "FTHG"] > df.loc[i, "FTHG"]:
        match_result = "Home"
    elif df.loc[i, "FTHG"] < df.loc[i, "FTHG"]:
        match_result = "Away"
    elif df.loc[i, "FTHG"] == df.loc[i, "FTHG"]:
        match_result ="Draw"
    
    if proba_home > proba_draw and proba_home > proba_away:
        forecast = "Home"
    elif proba_draw > proba_home and proba_draw > proba_away:
        forecast = "Draw"
    elif proba_away > proba_home and proba_away > proba_draw:
        forecast = "Away"
    
    if match_result == forecast:
        count+=1
print("Amount of times correct:", count)
print("Percentage correct:", (count/len(df)))