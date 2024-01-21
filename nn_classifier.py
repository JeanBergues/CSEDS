def train_neural_network_classifier(data: pd.DataFrame, y_name: str, no_hidden_layers: int, hidden_layer_size: int):
    # TODO: tweak other parameters in regressor function
    neural_network = nn.MLPClassifier(hidden_layer_sizes=(no_hidden_layers, hidden_layer_size))
    neural_network.fit(data.drop(y_name, axis=1), data[y_name])
    return neural_network

def main() -> None:
    # Train the neural network
    class_nn = train_neural_network_classifier(data.drop('Date', axis=1), 'FTR', no_hidden_layers, hidden_layer_size)

    # Predict one specific game
    print(team_mapping)
    pred_hometeam = team_mapping.get_loc('Ajax')
    pred_awayteam = team_mapping.get_loc('Nijmegen')
    pred_date = '22/01/2024'

    input_vec = pd.DataFrame(
        {
            'HomeTeam': pred_hometeam, 
            'AwayTeam': pred_awayteam,
            'ORDDate': datetime.strptime(pred_date, '%d/%m/%Y').date().toordinal(),
        }, index=[0])

    print(class_nn.predict(input_vec))
    print(data)

if __name__ == "__main__":
    main()