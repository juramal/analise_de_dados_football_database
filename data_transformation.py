import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "dataset/treated_dataset"
gameStats = pd.read_csv(f"{path}/gameStats.csv", encoding="latin1")
playerStatsinGame = pd.read_csv(f"{path}/playerStatsinGame.csv", encoding="latin1")
shotStats = pd.read_csv(f"{path}/shotStats.csv", encoding="latin1")
teamStats = pd.read_csv(f"{path}/teamStats.csv", encoding="latin1")

# 1. Transformar numérico para nominal
playerStatsinGame['positionOrder'] = playerStatsinGame['positionOrder'].map({
    1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"
})
playerStatsinGame['yellowCard'] = playerStatsinGame['yellowCard'].map({0: "Não", 1: "Sim"})
playerStatsinGame['redCard'] = playerStatsinGame['redCard'].map({0: "Não", 1: "Sim"})

# 2. Discretizar
gameStats['homeGoals'] = pd.cut(gameStats['homeGoals'], bins=[-1, 0, 1, float('inf')], labels=["Nenhum Gol", "Poucos Gols", "Muitos Gols"])
playerStatsinGame['time'] = pd.cut(playerStatsinGame['time'], bins=[-1, 30, 60, 90], labels=["Pouco Tempo", "Tempo Médio", "Muito Tempo"])

# 3. Normalizar
scaler = MinMaxScaler()
gameStats[['homeProbability', 'drawProbability', 'awayProbability']] = scaler.fit_transform(
    gameStats[['homeProbability', 'drawProbability', 'awayProbability']]
)
playerStatsinGame[['xGoals', 'xGoalsChain', 'xGoalsBuildup']] = scaler.fit_transform(
    playerStatsinGame[['xGoals', 'xGoalsChain', 'xGoalsBuildup']]
)

# Salvar os dados transformados
gameStats.to_csv("gameStats_transformed.csv", index=False)
playerStatsinGame.to_csv("playerStats_transformed.csv", index=False)