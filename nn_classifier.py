import pandas as pd
import numpy as np
import sklearn.neural_network as nn
from datetime import datetime as dt

def train_neural_network_classifier(data: pd.DataFrame, y_name: str, no_hidden_layers: int, hidden_layer_size: int):
    # TODO: tweak other parameters in regressor function
    neural_network = nn.MLPClassifier(hidden_layer_sizes=(no_hidden_layers, hidden_layer_size))
    neural_network.fit(data.drop(y_name, axis=1), data[y_name])
    return neural_network

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
    print(last_game)
    if last_game['HomeTeamID'] == htID:
        return transform_result_into_number(last_game['FTR'], htID, atID, htID)
    else:
        return transform_result_into_number(last_game['FTR'], atID, htID, htID)


def main() -> None:
    # Output from preprocessing
    starting_ordinal_date = 729978
    data = pd.read_csv('processed_data.csv')

    # NN parameters
    no_hidden_layers = 30
    hidden_layer_size = 100
    columns_to_use = ['FTR', 'DateNR', 'HomeTeamID', 'AwayTeamID']

    # Recreate data samples from Koopman
    training_data_cutoff = dt.strptime('31/07/09', '%d/%m/%y').date().toordinal() - starting_ordinal_date
    training_data = data[data['DateNR'] < training_data_cutoff]

    oss_data_cutoff = dt.strptime('11/08/17', '%d/%m/%y').date().toordinal() - starting_ordinal_date
    oos_data = data[data['DateNR'] < oss_data_cutoff]
    oos_data = oos_data[oos_data['DateNR'] >= training_data_cutoff]

    test_match = oos_data.iloc[300]
    
    # Train the neural network
    class_nn = train_neural_network_classifier(training_data[columns_to_use], 'FTR', no_hidden_layers, hidden_layer_size)

    # Predict the oos
    prediction = class_nn.predict(oos_data[columns_to_use].drop('FTR', axis=1))
    print(prediction)
    print(np.unique(prediction, return_counts=True))
    print(np.unique(oos_data['FTR'], return_counts=True))

    # Predict one specific game
    # print(team_mapping)
    # pred_hometeam = team_mapping.get_loc('Ajax')
    # pred_awayteam = team_mapping.get_loc('Nijmegen')
    # pred_date = '22/01/2024'

    # input_vec = pd.DataFrame(
    #     {
    #         'HomeTeam': pred_hometeam, 
    #         'AwayTeam': pred_awayteam,
    #         'ORDDate': datetime.strptime(pred_date, '%d/%m/%Y').date().toordinal(),
    #     }, index=[0])

    # print(class_nn.predict(input_vec))
    # print(data)

if __name__ == "__main__":
    main()