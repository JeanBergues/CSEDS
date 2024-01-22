import numpy as np
import math
import pandas as pd

def clean_data():
    # Reading all data
    data1 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data2 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data3 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data4 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data5 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data6 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data7 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data8 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data9 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data10 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data11 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data12 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data13 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")
    data14 = pd.read_csv("/Users/romnickevangelista/Documents/CSEDS/data_nl/season0001.csv")





    data["round"] = np.nan
    count = 0
    occured_teams = []
    for i in range(len(data)):
        if data["HomeTeam"][i] in occured_teams or data["AwayTeam"][i] in occured_teams:
            count += 1
            occured_teams = []
        else:
            occured_teams.append(data["HomeTeam"][i])
            occured_teams.append(data["AwayTeam"][i])
        data["round"][i] = count
    
    data.to_csv("Test1_data")

clean_data()