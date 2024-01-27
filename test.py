import bisect

# Existing data
teams = ['Team A', 'Team C', 'Team D']
f = [10, 15, 12, 8, 9, 11]  # Example values for attack and defense strengths

# New team's information
new_team_name = 'Team B'
new_team_attack = 13
new_team_defense = 10

# Calculate the insertion index for the new team's attack and defense values
insertion_index_attack = bisect.bisect(teams, new_team_name)  # Insert in sorted order
insertion_index_defense = insertion_index_attack + len(teams)+1

# Insert the new team's attack and defense values into f
f.insert(insertion_index_attack, new_team_attack)
f.insert(insertion_index_defense, new_team_defense)

# Append the new team's name to the teams list
teams.insert(insertion_index_attack, new_team_name)

# Verify the updated lists
print("Updated teams list:", teams)
print("Updated f list:", f)