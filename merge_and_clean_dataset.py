import pandas as pd
import os

playerStatsinGame = pd.read_csv("dataset/appearances.csv", encoding='latin-1')
gameStats = pd.read_csv("dataset/games.csv", encoding='latin-1')
leagueNames = pd.read_csv("dataset/leagues.csv", encoding='latin-1')
playerNames = pd.read_csv("dataset/players.csv", encoding='latin-1')
shotStats = pd.read_csv("dataset/shots.csv", encoding='latin-1')
teamNames = pd.read_csv("dataset/teams.csv", encoding='latin-1')
teamStats = pd.read_csv("dataset/teamstats.csv", encoding='latin-1')

# Substituir teamID pelo nome do time no df teamStats
teamStats['teamID'] = teamStats['teamID'].astype(int)
teamNames['teamID'] = teamNames['teamID'].astype(int)
mapping = teamNames.set_index('teamID')['name']
teamStats['teamID'] = teamStats['teamID'].map(mapping).fillna(teamStats['teamID'].astype(str))
teamStats = teamStats.rename(columns={'teamID': 'teamName'})

# Substituir playerID e leagueID no df playerStatsinGame pelos nomes correspondentes
playerStatsinGame['playerID'] = playerStatsinGame['playerID'].astype(int)
playerNames['playerID'] = playerNames['playerID'].astype(int)
mapping_players = playerNames.set_index('playerID')['name']
playerStatsinGame['playerID'] = playerStatsinGame['playerID'].map(mapping_players).fillna(playerStatsinGame['playerID'].astype(str))
playerStatsinGame = playerStatsinGame.rename(columns={'playerID': 'playerName'})

playerStatsinGame['leagueID'] = playerStatsinGame['leagueID'].astype(int)
leagueNames['leagueID'] = leagueNames['leagueID'].astype(int)
mapping_leagues = leagueNames.set_index('leagueID')['name']
playerStatsinGame['leagueID'] = playerStatsinGame['leagueID'].map(mapping_leagues).fillna(playerStatsinGame['leagueID'].astype(str))
playerStatsinGame = playerStatsinGame.rename(columns={'leagueID': 'leagueName'})

# Substituir shooterID e assisterID no df shotStats pelo nome correspondente
shotStats['shooterID'] = pd.to_numeric(shotStats['shooterID'], errors='coerce')
shotStats['assisterID'] = pd.to_numeric(shotStats['assisterID'], errors='coerce')
mapping_players = playerNames.set_index('playerID')['name']
shotStats['shooterID'] = shotStats['shooterID'].map(mapping_players)
shotStats['assisterID'] = shotStats['assisterID'].map(mapping_players)
shotStats = shotStats.rename(columns={'shooterID': 'shooterName', 'assisterID': 'assisterName'})

# Substituir leagueID, homeTeamID e awayTeamID no df gameStats pelos nomes correspondentes
gameStats['leagueID'] = gameStats['leagueID'].astype(int)
gameStats['homeTeamID'] = gameStats['homeTeamID'].astype(int)
gameStats['awayTeamID'] = gameStats['awayTeamID'].astype(int)
mapping_leagues = leagueNames.set_index('leagueID')['name']
mapping_teams = teamNames.set_index('teamID')['name']
gameStats['leagueID'] = gameStats['leagueID'].map(mapping_leagues).fillna(gameStats['leagueID'].astype(str))
gameStats['homeTeamID'] = gameStats['homeTeamID'].map(mapping_teams).fillna(gameStats['homeTeamID'].astype(str))
gameStats['awayTeamID'] = gameStats['awayTeamID'].map(mapping_teams).fillna(gameStats['awayTeamID'].astype(str))
gameStats = gameStats.rename(columns={'leagueID': 'leagueName', 'homeTeamID': 'homeTeamName', 'awayTeamID': 'awayTeamName'})

# Novos dfs tratados
output_folder = "dataset/treated_dataset"
os.makedirs(output_folder, exist_ok=True)
playerStatsinGame.to_csv(f"{output_folder}/playerStatsinGame.csv", index=False, encoding='latin-1')
gameStats.to_csv(f"{output_folder}/gameStats.csv", index=False, encoding='latin-1')
shotStats.to_csv(f"{output_folder}/shotStats.csv", index=False, encoding='latin-1')
teamStats.to_csv(f"{output_folder}/teamStats.csv", index=False, encoding='latin-1')