import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

path = "dataset/treated_dataset"
output_path = "dataset/transformed_dataset"
os.makedirs(output_path, exist_ok=True)

# Carregar os dados tratados
gameStats = pd.read_csv(f"{path}/gameStats.csv", encoding="latin1")
playerStatsinGame = pd.read_csv(f"{path}/playerStatsinGame.csv", encoding="latin1")
shotStats = pd.read_csv(f"{path}/shotStats.csv", encoding="latin1")
teamStats = pd.read_csv(f"{path}/teamStats.csv", encoding="latin1")

# DISCRETIZAÇÕES
gameStats['homeGoals'] = pd.cut(gameStats['homeGoals'], bins=[-1, 0, 1, float('inf')],
                                 labels=["No Goals", "Few Goals", "Many Goals"])
gameStats['awayGoals'] = pd.cut(gameStats['awayGoals'], bins=[-1, 0, 1, float('inf')],
                                 labels=["No Goals", "Few Goals", "Many Goals"])
gameStats['homeGoalsHalfTime'] = pd.cut(gameStats['homeGoalsHalfTime'], bins=[-1, 0, 1, float('inf')],
                                 labels=["0", "1", "2+"])
gameStats['awayGoalsHalfTime'] = pd.cut(gameStats['awayGoalsHalfTime'], bins=[-1, 0, 1, float('inf')],
                                 labels=["0", "1", "2+"])
shotStats['minute'] = pd.cut(shotStats['minute'], bins=[-1, 15, 30, 45, 60, 75, 90],
                              labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76-90"])
teamStats['pressure'] = pd.cut(teamStats['ppda'], bins=[-1, 7, 12, float('inf')],
                               labels=["High Pressure", "Medium Pressure", "Low Pressure"])

# NORMALIZAÇÕES (MinMaxScaler)
scaler = MinMaxScaler()

gameStats[['homeProbability', 'drawProbability', 'awayProbability']] = scaler.fit_transform(
    gameStats[['homeProbability', 'drawProbability', 'awayProbability']])

playerStatsinGame[['xGoals', 'xGoalsChain', 'xGoalsBuildup', 'xAssists', 'shots']] = scaler.fit_transform(
    playerStatsinGame[['xGoals', 'xGoalsChain', 'xGoalsBuildup', 'xAssists', 'shots']])

shotStats[['xGoal', 'positionX', 'positionY']] = scaler.fit_transform(
    shotStats[['xGoal', 'positionX', 'positionY']])

teamStats[['shots', 'shotsOnTarget', 'goals', 'xGoals', 'deep']] = scaler.fit_transform(
    teamStats[['shots', 'shotsOnTarget', 'goals', 'xGoals', 'deep']])

gameStats.to_csv(f"{output_path}/gameStats_transformed.csv", index=False)
playerStatsinGame.to_csv(f"{output_path}/playerStatsinGame_transformed.csv", index=False)
shotStats.to_csv(f"{output_path}/shotStats_transformed.csv", index=False)
teamStats.to_csv(f"{output_path}/teamStats_transformed.csv", index=False)
