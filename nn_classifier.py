import pandas as pd
import numpy as np
import sklearn.neural_network as nn
from datetime import datetime as dt

def train_neural_network_classifier(data: pd.DataFrame, y_name: str, hidden_layer_sizes):
    # TODO: tweak other parameters in regressor function
    neural_network = nn.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=1234, max_iter=300)
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
    if last_game['HomeTeamID'] == htID:
        return transform_result_into_number(last_game['FTR'], htID, atID, htID)
    else:
        return transform_result_into_number(last_game['FTR'], atID, htID, htID)
    

def calculate_succes_ratio(realizations, forecast):
    return 1 - np.count_nonzero(realizations - forecast) / len(realizations)


def output_type_errors(realizations, forecast):
    errors = np.zeros(9, dtype=np.int16)
    process_count = 0
    for r, f in zip(realizations, forecast):
        if r == -1: # Realized loss
            process_count += 1
            errors[1 + f] = errors[1 + f] + 1
        elif r == 0:# Realized draw
            process_count += 1
            errors[4 + f] = errors[4 + f] + 1
        elif r == 1:# Realized win
            process_count += 1
            errors[7 + f] = errors[7 + f] + 1

    return errors.reshape(3, 3)

def main() -> None:
    # Output from preprocessing
    starting_ordinal_date = 729978
    data = pd.read_csv('processed_data.csv')

    # Remap FTR
    ftr_mapping = {'A': -1, 'D': 0, 'H': 1}
    data['FTR'] = data['FTR'].apply(lambda x: ftr_mapping[x])

    # Rescale categorials
    max_team = max(data['HomeTeamID'].max(), data['AwayTeamID'].max())
    data['HomeTeamID'] = data['HomeTeamID'].apply(lambda x: x/max_team)
    data['AwayTeamID'] = data['AwayTeamID'].apply(lambda x: x/max_team)

    # NN parameters
    hidden_layer_sizes = np.repeat(5, 2)
    hidden_layer_sizes = [30, 20, 10]
    columns_to_use = ['FTR', 'HomeTeamID', 'AwayTeamID','PrevHTR','PrevATR','PrevDR']
    # columns_to_use = ['FTR', 'PrevHTR','PrevATR','PrevDR']

    # Recreate data samples from Koopman
    training_data_cutoff = dt.strptime('31/07/09', '%d/%m/%y').date().toordinal() - starting_ordinal_date
    training_data = data[data['DateNR'] < training_data_cutoff]

    oss_data_cutoff = dt.strptime('11/08/17', '%d/%m/%y').date().toordinal() - starting_ordinal_date
    oos_data = data[data['DateNR'] < oss_data_cutoff]
    oos_data = oos_data[oos_data['DateNR'] >= training_data_cutoff]

    # max_time = data['DateNR'].max()
    # data['DateNR'] = data['DateNR'].apply(lambda x: x/max_time)
    # print(data)
    
    # Train the neural network
    class_nn = train_neural_network_classifier(training_data[columns_to_use], 'FTR', hidden_layer_sizes)

    # Predict the oos
    prediction = class_nn.predict(training_data[columns_to_use].drop('FTR', axis=1))
    realization = training_data['FTR']

    # Analyze the results
    print(np.unique(prediction, return_counts=True))
    print(np.unique(realization, return_counts=True))

    succes_ratio = calculate_succes_ratio(np.array(realization), prediction)
    print(f"Succes ratio: {succes_ratio:.3f}%")
    MSE = np.sum(np.square(realization - prediction))
    print(f"MSE: {MSE:d}")
    type_errors = output_type_errors(np.array(realization), prediction)
    print(type_errors)


    # prediction = np.zeros(N)
    # for i in range(N):
    #     current_row = oos_data.iloc[[i]]
    #     oos_data.iloc[[i]]['PrevHTR'] = 2
    #     prediction[i] = class_nn.predict(oos_data.iloc[[i]][columns_to_use].drop('FTR', axis=1))

    # print(prediction)
    # succes_ratio = 1 - np.count_nonzero(np.array(oos_data['FTR']) - prediction) / N
    # print(succes_ratio)

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