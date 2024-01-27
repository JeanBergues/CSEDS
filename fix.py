# For fixing f order
est = pd.read_csv("BP_results_for_NN.csv")
f = literal_eval(est["f"][0])
teams = schedule["HomeTeam"].unique().tolist()
# Zip the teams and values together
combined = zip(teams, f[:len(teams)], f[len(teams):])
# Sort the zipped pairs based on team names
sorted_combined = sorted(combined, key=lambda x: x[0])
# Extract the sorted values
sorted_attack_strengths = [strength[1] for strength in sorted_combined]
sorted_defense_strengths = [strength[2] for strength in sorted_combined]

# Concatenate the sorted attack and defense strengths
sorted_f = sorted_attack_strengths + sorted_defense_strengths

df = pd.DataFrame({"a1": [est["a1"][0]],
                   "a2": [est["a2"][0]],
                    "b1": [est["b1"][0]],
                    "b2": [est["b2"][0]],
                    "lambda3": [est["lambda3"][0]],
                    "delta": [est["delta"][0]],
                    "f": [sorted_f]})
df.to_csv("BP_results_for_NN_fixed.csv", index=False)