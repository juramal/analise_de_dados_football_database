import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "dataset/treated_dataset"
gameStats = pd.read_csv(f"{path}/gameStats.csv", encoding="latin1")
playerStatsinGame = pd.read_csv(f"{path}/playerStatsinGame.csv", encoding="latin1")
shotStats = pd.read_csv(f"{path}/shotStats.csv", encoding="latin1")
teamStats = pd.read_csv(f"{path}/teamStats.csv", encoding="latin1")

def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def create_boxplot_and_find_outliers(df, columns, title):
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot for {column} - {title}')
            plt.show()
            
            outliers = find_outliers_iqr(df, column)
            if not outliers.empty:
                print(f"Outliers detected in column '{column}' - {title}:")
                print(outliers[[column]])
            else:
                print(f"No outliers detected in column '{column}' - {title}.")

# Separar apenas colunas numéricas
shot_stats_columns = shotStats.select_dtypes(include=['float64', 'int64']).columns
game_stats_columns = gameStats.select_dtypes(include=['float64', 'int64']).columns
player_stats_columns = playerStatsinGame.select_dtypes(include=['float64', 'int64']).columns
team_stats_columns = teamStats.select_dtypes(include=['float64', 'int64']).columns

print("Analyzing shotStats.csv")
create_boxplot_and_find_outliers(shotStats, shot_stats_columns, "shotStats")

print("\nAnalyzing gameStats.csv")
create_boxplot_and_find_outliers(gameStats, game_stats_columns, "gameStats")

print("\nAnalyzing playerStatsinGame.csv")
create_boxplot_and_find_outliers(playerStatsinGame, player_stats_columns, "playerStatsinGame")

print("\nAnalyzing teamStats.csv")
create_boxplot_and_find_outliers(teamStats, team_stats_columns, "teamStats")