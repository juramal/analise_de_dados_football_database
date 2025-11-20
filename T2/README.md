# T2 - Classificadores de Relação Jogadores e Chutes

## Objetivo

Classificar a probabilidade de um chute resultar em gol com base nas características do jogador, do chute e do contexto do jogo, utilizando os dados completos do diretório `transformed_dataset`.

## Estrutura

```
T2/
├── classifiers/
│   ├── NaiveBayes.py           # Classificador Naive Bayes
│   ├── J48.py                  # Classificador J48 (Decision Tree C4.5)
│   ├── RandomForest.py         # Classificador Random Forest
│   └── MultilayerPerceptron.py # Classificador MLP (Rede Neural)
├── results/                    # Resultados gerados pelos classificadores
└── run_all_classifiers.py      # Script para executar todos os classificadores
```

## Datasets Utilizados

Todos os classificadores utilizam os seguintes datasets:

- **shotStats_transformed.csv**: Estatísticas de chutes (324,545 registros)
  - Colunas: gameID, shooterName, assisterName, minute, situation, lastAction, shotType, shotResult, xGoal, positionX, positionY

- **playerStatsinGame_transformed.csv**: Estatísticas de jogadores por partida (356,515 registros)
  - Colunas: gameID, playerName, goals, ownGoals, shots, xGoals, xGoalsChain, xGoalsBuildup, assists, keyPasses, xAssists, position, positionOrder, yellowCard, redCard, time, substituteIn, substituteOut, leagueName

- **teamStats_transformed.csv**: Estatísticas de times por partida (25,362 registros)
  - Colunas: gameID, teamName, season, date, location, goals, xGoals, shots, shotsOnTarget, deep, ppda, fouls, corners, yellowCards, redCards, result, pressure

- **gameStats_transformed.csv**: Estatísticas de jogos (12,682 registros)
  - Colunas: gameID, leagueName, season, date, homeTeamName, awayTeamName, homeGoals, awayGoals, homeProbability, drawProbability, awayProbability, homeGoalsHalfTime, awayGoalsHalfTime, odds...

## Classificadores

### 1. Naive Bayes (`NaiveBayes.py`)

**Algoritmo**: Gaussian Naive Bayes

**Características**:
- Baseado no teorema de Bayes com suposição de independência entre features
- Rápido e eficiente para grandes datasets
- Bom baseline para problemas de classificação

**Saídas**:
- `naivebayes_results.csv`: Métricas de desempenho
- `naivebayes_confusion_matrix.png`: Matriz de confusão
- `naivebayes_feature_importance.png`: Importância das features

### 2. J48 (`J48.py`)

**Algoritmo**: Decision Tree (implementação similar ao C4.5 do Weka)

**Características**:
- Usa ganho de informação (entropy) para dividir nós
- Cria regras interpretáveis
- Visualização da árvore de decisão

**Saídas**:
- `j48_results.csv`: Métricas de desempenho
- `j48_confusion_matrix.png`: Matriz de confusão
- `j48_feature_importance.png`: Importância das features
- `j48_tree_visualization.png`: Visualização da árvore (3 níveis)

### 3. Random Forest (`RandomForest.py`)

**Algoritmo**: Ensemble de múltiplas árvores de decisão

**Características**:
- 100 árvores de decisão
- Votação por maioria para classificação
- Robusto contra overfitting
- Calcula AUC e ROC curve

**Saídas**:
- `randomforest_results.csv`: Métricas de desempenho
- `randomforest_confusion_matrix.png`: Matriz de confusão
- `randomforest_feature_importance.png`: Importância das features
- `randomforest_feature_importance.csv`: Importância em CSV
- `randomforest_roc_curve.png`: Curva ROC

### 4. Multilayer Perceptron (`MultilayerPerceptron.py`)

**Algoritmo**: Rede Neural Artificial

**Características**:
- Arquitetura: 100-50-25 neurônios (3 camadas ocultas)
- Função de ativação: ReLU
- Otimizador: Adam
- Early stopping para evitar overfitting

**Saídas**:
- `mlp_results.csv`: Métricas de desempenho
- `mlp_confusion_matrix.png`: Matriz de confusão
- `mlp_learning_curve.png`: Curva de aprendizado (loss)

## Como Executar

### Executar um classificador específico:

```powershell
python T2/classifiers/NaiveBayes.py
python T2/classifiers/J48.py
python T2/classifiers/RandomForest.py
python T2/classifiers/MultilayerPerceptron.py
```

### Executar todos os classificadores:

```powershell
python T2/run_all_classifiers.py
```

Este script:
1. Executa todos os 4 classificadores sequencialmente
2. Compara os resultados
3. Gera gráficos comparativos
4. Cria relatório resumido

## Features Utilizadas

Cada classificador utiliza as seguintes features:

**Características do Chute**:
- `xGoal`: Expected Goals do chute
- `positionX`, `positionY`: Posição do chute no campo
- `minute`: Minuto do jogo
- `shotType_encoded`: Tipo de chute (encoding)
- `situation_encoded`: Situação do chute (encoding)
- `lastAction_encoded`: Última ação antes do chute (encoding)

**Características do Jogador**:
- `goals`: Gols marcados na partida
- `shots`: Chutes na partida
- `xGoals`: Expected Goals do jogador
- `assists`: Assistências
- `keyPasses`: Passes-chave
- `position_encoded`: Posição do jogador (encoding)

**Características do Time**:
- `pressure_encoded`: Nível de pressão do time (encoding)

## Variável Target

**`shotResult`**: Classificação binária
- `1`: Goal (gol marcado)
- `0`: No Goal (chute não resultou em gol)

## Métricas de Avaliação

Todos os classificadores calculam:

- **Accuracy**: Acurácia geral do modelo
- **Precision**: Precisão por classe
- **Recall**: Revocação por classe
- **F1-Score**: Média harmônica entre precisão e recall
- **Cross-Validation**: Validação cruzada (5-fold)
- **Confusion Matrix**: Matriz de confusão

Random Forest também calcula:
- **AUC**: Area Under the Curve
- **ROC Curve**: Receiver Operating Characteristic

## Resultados Esperados

Após a execução, os seguintes arquivos serão gerados em `T2/results/`:

1. Resultados individuais de cada modelo (CSV)
2. Matrizes de confusão (PNG)
3. Importância das features (PNG e CSV)
4. Comparação entre todos os modelos (`comparison_all_classifiers.csv`)
5. Gráfico comparativo (`comparison_chart.png`)

## Observações

- Os dados são divididos em 70% treino e 30% teste
- Stratified split mantém proporção das classes
- Features categóricas são encodadas usando LabelEncoder
- MLP normaliza os dados usando StandardScaler
- Valores nulos são preenchidos com 0

## Requisitos

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Instalar com:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn
```
