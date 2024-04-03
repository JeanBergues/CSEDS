def get_f(data, schedule, params):
    # Define parameters
    a1, a2, b1, b2, lambda3, delta, *f_ini = params

    # Length of data
    T = len(data)

    # Setting log likelihood to 0 first
    ll = 0

    # Get list of distinct teams
    all_teams = schedule['HomeTeam'].unique().tolist()
    nr_teams = len(all_teams)
    
    # Defining f_t
    f = []

    # Calculating f_t
    for t in range(T):
        sum_1 = 0
        if t == 0:
            f.append(f_ini)
            
            # Get all matches from round
            schedule_round = schedule[schedule['round'] == t]

            # Create w
            B_all_teams = create_A_B_matrix(b1,b2, nr_teams)
            w = np.multiply(f[t], (np.ones(len(f[t])) - np.diagonal(B_all_teams)))

            # Create empty list for f_t+1
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round["HomeTeam"][i]
                away = schedule_round["AwayTeam"][i]

                # Get attack strength and defense strength of specific teams from f_t
                # The order of f_t is same as unique list
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away

            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season
            
        else:
            empty_list = np.empty(len(f[0]))
            empty_list[:] = np.nan
            f.append(empty_list)

            schedule_round = schedule[schedule['round'] == t]
            for i in range(len(schedule_round)):
                # Get match opponents
                home = schedule_round.iloc[i]["HomeTeam"]
                away = schedule_round.iloc[i]["AwayTeam"]

                # Get index of teams
                home_index = all_teams.index(home)
                away_index = all_teams.index(away)

                # Get corresponding data
                x = data.iloc[t][home]
                y = data.iloc[t][away]

                # get previous lambda1 and lambda2
                prev_alpha_home = f[t][home_index]
                prev_alpha_away = f[t][away_index]
                prev_beta_home = f[t][home_index + nr_teams]
                prev_beta_away = f[t][away_index + nr_teams]

                # Calc s_t
                s = calc_s(x, y, prev_alpha_home, prev_alpha_away, prev_beta_home, prev_beta_away, lambda3, delta)

                # Update f_t
                f[t+1][home_index] = w[home_index] + b1 * f[t][home_index] + a1 * s[0] # Attack strength home
                f[t+1][away_index] = w[away_index] + b1 * f[t][away_index] + a1 * s[1] # Attack strength away
                f[t+1][home_index + nr_teams] = w[home_index + nr_teams] + b2 * f[t][home_index + nr_teams] + a2 * s[2] # Defense strength home
                f[t+1][away_index + nr_teams] = w[away_index + nr_teams] + b2 * f[t][away_index + nr_teams] + a2 * s[3] # Defense strength away
                
            # Filling in f_t when team does not play
            index_no_play = np.where(np.isnan(f[t+1]))[0]
            teams_no_play = [i for i in index_no_play if i < len(all_teams)]

            for i in teams_no_play:
                f[t+1][i] = w[i] + b1 * f[t][i]
                f[t+1][i + len(all_teams)] = w[i + len(all_teams)] + b2 * f[t][i + len(all_teams)]
                # We dont update ll for these team since no opponent
                # Eventually it will update since there needs to be certain matches a season

    return f