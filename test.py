import numpy as np 
import pandas as pd

df = pd.read_csv("BP_ONE_SEASON_FIX_NEW.csv")
df = df.reset_index()
count = 0
df["Prediction_num"]
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
    
    if forecast == "Home":
        df.loc[i, "Prediction_num"] = 0
    if forecast == "Away":
        df.loc[i, "Prediction_num"] = 2
    if forecast == "Draw":
        df.loc[i, "Prediction_num"] = 1
df.to_csv("Final_Forecasts.csv", index=False)
print("Amount of times correct:", count)
print("Percentage correct:", (count/len(df)))