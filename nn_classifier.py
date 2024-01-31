import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.preprocessing as prep
import sklearn.model_selection as modsel
import sklearn.metrics as met
import itertools
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
        if r == 0: # Realized loss
            process_count += 1
            errors[0 + f] = errors[0 + f] + 1
        elif r == 1:# Realized draw
            process_count += 1
            errors[3 + f] = errors[3 + f] + 1
        elif r == 2:# Realized win
            process_count += 1
            errors[6 + f] = errors[6 + f] + 1

    return errors.reshape(3, 3)

def main() -> None:
    # Output from preprocessing
    starting_ordinal_date = 729978
    INCLUDE_ATTACK_DEFENSE = True
    data = pd.read_csv('schedule_for_NN_FIX.csv')
    data = pd.read_csv('processed_data.csv')
    
    data.drop(data.columns[data.columns.str.contains('unnamed',case = False)], axis = 1, inplace = True)

    # Remap FTR
    ftr_mapping = {'A': 0, 'D': 1, 'H': 2}
    data['FTR'] = data['FTR'].apply(lambda x: ftr_mapping[x])

    # Rescale categorials
    # max_team = max(data['HomeTeamID'].max(), data['AwayTeamID'].max())
    # data['HomeTeamID'] = data['HomeTeamID'].apply(lambda x: x/max_team)
    # data['AwayTeamID'] = data['AwayTeamID'].apply(lambda x: x/max_team)

    max_points = max(data['HomePrevSeasonPoints'].max(), data['AwayPrevSeasonPoints'].max())
    data['HomePrevSeasonPoints'] = data['HomePrevSeasonPoints'].apply(lambda x: x/max_points)
    data['AwayPrevSeasonPoints'] = data['AwayPrevSeasonPoints'].apply(lambda x: x/max_points)

    # Rescale season results
    data['HomePrevSeasonPos'] = data['HomePrevSeasonPos'].apply(lambda x: x/18)
    data['AwayPrevSeasonPos'] = data['AwayPrevSeasonPos'].apply(lambda x: x/18)

    # Apply column selection
    experiment = 5
    experiments = [
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'HomeAttack', 'HomeDefense', 'AwayAttack', 'AwayDefense'],
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'HomePrevSeasonPos', 'AwayPrevSeasonPos'],
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'HomePrevSeasonPos', 'HomePrevSeasonPoints', 'AwayPrevSeasonPos', 'AwayPrevSeasonPoints'],
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'HomeAttack', 'HomeDefense', 'AwayAttack', 'AwayDefense', 'HomePrevSeasonPos', 'HomePrevSeasonPoints', 'AwayPrevSeasonPos', 'AwayPrevSeasonPoints'],
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'PrevHTR', 'PrevATR', 'PrevDR'],
        ['FTR', 'Season', 'HomeTeamID', 'AwayTeamID', 'PowerH', 'PowerA'],
    ]

    data = pd.read_csv('probit_full_NN.csv')
    data['FTR'] = data['FTR'].apply(lambda x: int(x))
    columns_to_use = experiments[experiment]
    exclude_info = ['Season', 'HomeTeamID','AwayTeamID']
    data = data[columns_to_use]
    # data = pd.get_dummies(data=data, columns=['HomeTeamID', 'AwayTeamID'])
    # print(o_data)

    # Split data
    training_data = data[(data.Season >= 1) & (data.Season < 20)].drop(exclude_info, axis=1)
    oos_data = data[data.Season >= 20].drop(exclude_info, axis=1)

    # oss_data_cutoff = dt.strptime('11/08/17', '%d/%m/%y').date().toordinal() - starting_ordinal_date
    # oos_data = data[data['DateNR'] < oss_data_cutoff]
    # oos_data = oos_data[oos_data['DateNR'] >= training_data_cutoff]

    # max_time = data['DateNR'].max()
    # data['DateNR'] = data['DateNR'].apply(lambda x: x/max_time)
    # print(data)
    
    # Train the neural network
    realization = training_data['FTR']
    APPLY_GRID_SEARCH = False
    if APPLY_GRID_SEARCH:
        option_neurons = [2, 5, 10, 20]
        options = [x for x in itertools.product(option_neurons, repeat=2)]
        options.extend([x for x in option_neurons])
        best_option = [1, 1]
        best_criterion = np.inf

        # for option in options:
        #     print(f"Testing {option}")
        #     class_nn = train_neural_network_classifier(training_data.drop(columns_to_not_use, axis=1), 'FTR', option)
        #     prediction = class_nn.predict(training_data.drop(columns_to_not_use, axis=1).drop('FTR', axis=1))
        #     criterion = met.mean_squared_error(realization, prediction)
        #     print(f"MSE of model: {criterion}")

        #     if criterion < best_criterion:
        #         print("Update!")
        #         best_criterion = criterion
        #         best_option = [option[0], option[1]]

        #     print(f"Best option: {best_option}")
        #     print(f"With MSE: {best_criterion}")

        gsearch_model = modsel.GridSearchCV(nn.MLPClassifier(), param_grid={'hidden_layer_sizes': options, 'random_state': [1234], 'max_iter': [300]}, n_jobs=-1, refit=True, verbose=3)
        gmodel = gsearch_model.fit(training_data.drop('FTR', axis=1), training_data['FTR'])

        class_nn = gmodel.best_estimator_
        chosen_layers = gmodel.best_params_['hidden_layer_sizes']
    else:
        chosen_layers = (50, 20)
        class_nn = train_neural_network_classifier(training_data, 'FTR', chosen_layers)
    
    print(f"Best layer structure: {chosen_layers}")

    

    # In-sample performance
    print(f"IN SAMPLE PERFORMANCE")
    prediction = class_nn.predict(training_data.drop('FTR', axis=1))
    realization = training_data['FTR']

    # Analyze the results
    print(np.unique(prediction, return_counts=True))
    print(np.unique(realization, return_counts=True))

    succes_ratio = calculate_succes_ratio(np.array(realization), prediction)
    print(f"Succes ratio: {succes_ratio:.3f}%")
    MSE = np.sum(np.square(realization - prediction))
    print(f"MSE: {MSE}")
    type_errors = output_type_errors(np.array(realization), prediction)
    print(type_errors)

    # Predict the oos
    print(f"OUT OF SAMPLE PERFORMANCE")
    prediction = class_nn.predict(oos_data.drop('FTR', axis=1))
    realization = oos_data['FTR']

    # Analyze the results
    print(np.unique(prediction, return_counts=True))
    print(np.unique(realization, return_counts=True))

    succes_ratio = calculate_succes_ratio(np.array(realization), prediction)
    print(f"Succes ratio: {succes_ratio:.3f}%")
    MSE = np.sum(np.square(realization - prediction))
    print(f"MSE: {MSE:d}")
    type_errors = output_type_errors(np.array(realization), prediction)
    print(type_errors)


    if len(chosen_layers) == 1:
        with open(f'predictions/nn_class_{chosen_layers[0]}_0_{"AD" if INCLUDE_ATTACK_DEFENSE else ""}', 'wb') as file:
            np.array(prediction).dump(file)   
    else:
        with open(f'predictions/nn_class_{chosen_layers[0]}_{chosen_layers[1]}_{"AD" if INCLUDE_ATTACK_DEFENSE else ""}', 'wb') as file:
            np.array(prediction).dump(file)

    proba_predictions = class_nn.predict_proba(oos_data.drop('FTR', axis=1))

    results = pd.DataFrame()
    results['Outcome'] = realization
    results['Prediction'] = prediction
    results['ProbA'] = proba_predictions[:,0]
    results['ProbD'] = proba_predictions[:,1]
    results['ProbH'] = proba_predictions[:,2]

    results.to_csv(f'predictions/df_nn_class_{chosen_layers[0]}_{chosen_layers[1]}_{"AD" if INCLUDE_ATTACK_DEFENSE else ""}.csv')

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