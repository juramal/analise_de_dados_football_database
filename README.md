# ⚽ Análise de dados no futebol

#### 🔗 Link da base de dados: https://www.kaggle.com/datasets/technika148/football-database?resource=download

Base de dados sobre futebol dos jogadores e times das top5 ligas europeias de 2014-2020, contendo informações valiosas para uma futura análise de dados, separadas em CSVs como se fosse um banco de dados relacional, sendo eles Appearances (Aparições), Games (jogos), Leagues (Ligas), Players (Jogadores), Shots (Chutes), Teams (Times) e TeamsStats (Estatísticas do time). As tabelas estão relacionadas através de ID como chave estangeira.

Este database está bem completo, podendo entregar dados interessantes para serem relacionados. como dados dos jogadores em partidas na tabela Shots, que há ShotType, positionX, positionY, xGoal (Expected Goal, ou Gol esperado, que é a porcentagem que aquele chute tem de ser gol, por exemplo, um chute de fora da área efetuada por um zagueiro tem menor xGoal que um chute na pequena área de um Atacante com gols no campeonato). Há mais desse tipo de informaão na tabela Appearances, que contribui para o database com dados mais gerais sobre as aparições em partidas de jogadores.

Há também informações pertinentes dos times, na tabela Games, como time da casa e time visitante, gols da partida, probabilidade de vitória segundo diferentes casas de apostas (que já utilizam análsie de dados para poder inferir esta informação) e informações específicas dos times na tabela TeamStats, como gols, xGoal, chutes, chutes no alvo, cartões, faltas, escanteios, deep (ou profundidade, que mede a quantidade de passes completos dentro da distância de 20 jardas do gol adversário, ou seja, passes completados perto do gol adiversário). Dados estes que podem ser utilizados para explicitar o modelo de ataque de uma equipe, uma provável correlação com os jogadores daquela equipe, se é um time eficiente ou não quando tem a posse de bola etc.

Apesar de ter muitas informações relevantes nesse dataset, ele está obviamente mais inclinado a relacionar os jogadores e times com os chutes e xGoal dos jogadores e times, devido a quantidade de dados nesse sentido.

---

### 📂 Estrutura do Projeto

| Diretório / Arquivo | Descrição |
|----------------------|------------|
| `/dataset/` | Contém os arquivos CSV originais (appearances, games, leagues, players, shots, teams, teamstats) |
| `venv` | Ambiente virtual python (versão 3.10) |
| `/dataset/treated_dataset` | Contém os arquivos CSV após a limpeza (gameStats, playerStatsinGame, shotStats e teamStats) |
| `/boxsplot_and_outliers.py/` | Script responsável por criar os gráficos boxsplot e identificação de outliers |
| `/data_transformation.py/` | Script que contém discretização, mudança de numérico para nominal e normalização dos dataframes |
| `/merge_and_clean_dataset.py/` | Limpeza do dataset com remoção de IDs por nomes (leagueID->leagueName, playerID->playerName etc.) |
| `README.md` | Este arquivo explicativo do projeto |
| `requirements.txt` | Arquivo de dependências para executar os scripts |

---

## 🧠 Tarefas do Projeto

<details>
<summary><b>🔗 VER TAREFA 1 – PROCESSO KDD E CLUSTERIZAÇÃO</b></summary>

## 🧩 Etapas do Processo KDD

O projeto segue as seguintes etapas do processo de **KDD**:

1. **Seleção de Dados** – Escolha dos datasets relevantes

    Inicialmente a base de dados tinha 7 tabelas que foram construídas em uma lógica de banco 
    de dados relacional, por isso, em todos os arquivos csvs há IDs que atuam como chaves 
    primárias e secundárias no relacionamento entre tabelas, a exemplo do leagueID que
    aparece nas tabelas games.csv e appearances.csv, para identificar a liga dos jogos e das 
    aparições dos jogadores.

### 📊 Sobre a base de dados

  #### 🧾 Tabela: `appearances`
  <details>
  <summary><b>Ver atributos de apperances</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | gameID | categórico | Identificador do jogo |
  | payerID | categórico | Identificador do jogador |
  | goals | numérico | Número de gols marcados |
  | ownGoals | numérico | Número de gols contra |
  | shots | numérico | Número de chutes do jogador |
  | xGoals | numérico | Probabilidade de um chute resultar em gol |
  | xGoalsChain | numérico | Posse de bola que resultou em um chute |
  | xGoalsBuildup | numérico | Contribuição do jogador para um chute |
  | assists | numérico | numérico de assistências |
  | keyPasses | numérico | Passe final antes de um chute |
  | xAssists | numérico | Probabilidade de uma assistência resultar em gol |
  | position | categórico | Posição do jogador em campo |
  | positionOrder | numérico | Ordem da posição em campo |
  | yellowCard | numérico | Cartões amarelos recebidos |
  | redCard | numérico | Cartões vermelhos recebidos |
  | time | numérico | Minutos jogados |
  | substituteIn | numérico | Jogador entrou em campo |
  | substituteOut | numérico | Jogador saiu de campo |
  | leagueID | categórico | Identificador da liga |

  </details>

  #### 🏟️ Tabela: `games`
  <details>
  <summary><b>Ver atributos de games</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | gameID | categórico | Identificador do jogo |
  | leagueID | categórico | Identificador da liga |
  | season | categórico | Ano da temporada |
  | date | temporal | Data e hora do jogo |
  | homeTeamID | categórico | Identificador do time da casa |
  | awayTeamID | categórico | Identificador do time visitante |
  | homeGoals | numérico | Gols do time da casa |
  | awayGoals | numérico | Gols do time visitante |
  | homeProbability | numérico | Probabilidade de vitória do time da casa |
  | drawProbability | numérico | Probabilidade de empate |
  | awayProbability | numérico | Probabilidade de vitória do time visitante |
  | homeGoalsHalfTime | numérico | Gols do time da casa no intervalo |
  | awayGoalsHalfTime | numérico | Gols do time visitante no intervalo |
  | B365H | numérico | Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta Bet365 |
  | B365D | numérico | Possibilidade de EMPATE segundo a casa de aposta Bet365 |
  | B365A | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta Bet365 |
  | BWH | numérico | Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta BW |
  | BWD | numérico | Possibilidade de EMPATE segundo a casa de aposta BW |
  | BWA | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta BW |
  | IWH | numérico | Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta IW |
  | IWD | numérico | Possibilidade de EMPATE segundo a casa de aposta BW |
  | IWA | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta IW |
  | PSH | numérico | Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta PS |
  | PSD | numérico | Possibilidade de EMPATE segundo a casa de aposta PS |
  | PSA | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta PS |
  | WHH | numérico |Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta WH |
  | WHD | numérico | Possibilidade de EMPATE segundo a casa de aposta WH |
  | WHA | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta WH |
  | VCH | numérico | Possibilidade de VITÓRIA do time da CASA segundo a casa de aposta VC |
  | VCD | numérico | Possibilidade de EMPATE segundo a casa de aposta VC |
  | VCA | numérico | Possibilidade de VITÓRIA do time da VISITANTE segundo a casa de aposta VC |

  </details>

  #### 🏆 Tabela: `leagues`
  <details>
  <summary><b>Ver atributos de leagues</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | leagueID | int | Identificador da liga |
  | name | categórico | Nome da liga |
  | understatNotation | categórico | Sigla de identificação da liga |

  </details>

  #### 👟 Tabela: `players`
  <details>
  <summary><b>Ver Atributos de players</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | playerID | categórico | Identificador do jogador |
  | name | categórico | Nome do jogador |

  </details>

  #### 🎯 Tabela: `shots`
  <details>
  <summary><b>Ver atributos de shots</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | gameID | categórico | Identificador do jogo |
  | shooterID | categórico | Jogador que realizou o chute |
  | assisterID | categórico | Jogador que deu assistência |
  | minute | numérico | Minuto do jogo do chute |
  | situation | categórico | Tipo de lance (ex: bola parada, contra-ataque) |
  | lastAction | categórico | Tipo da jogada anterior |
  | shotType | categórico | Pé utilizado no chute (esquerdo/direito) |
  | shotResult | categórico | Resultado do chute (gol, fora, bloqueado, etc.) |
  | xGoal | numérico | Probabilidade de um chute resultar em gol |
  | positionX | numérico | Coordenada X do chute |
  | positionY | numérico | Coordenada Y do chute |

  </details>

  #### 🛡️ Tabela: `teams`
  <details>
  <summary><b>Ver atributos de teams</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | teamID | categórico | Identificador do time |
  | name | categórico | Nome do time |

  </details>

  #### 📈 Tabela: `teamstats`
  <details>
  <summary><b>Ver atributos de teamstats</b></summary>

  | Atributo | Tipo | Descrição |
  |-----------|------|------------|
  | gameID | categórico | Identificador do jogo |
  | teamID | categórico | Identificador do time |
  | season | categórico | Ano da temporada |
  | date | temporal | Data e hora do jogo |
  | location | categórico | “h” para time da casa, “a” para visitante |
  | goals | numérico | Gols marcados pelo time |
  | xGoals | numérico | Probabilidade de um chute resultar em gol |
  | shots | numérico | Total de chutes realizados |
  | shotsOnTarget | numérico | Chutes no gol |
  | deep | numérico | Lances de fundo de área |
  | ppda | numérico | Índice de retomada de bola |
  | fouls | numérico | Faltas cometidas |
  | corners | numérico | Escanteios recebidos |
  | yellowCards | numérico | Cartões amarelos |
  | redCards | numérico | Cartões vermelhos |
  | result | categórico | Resultado do jogo (“W”, “L”, “D”) |

  </details>

<br>

---

<br>

2a. **Pré-processamento e Limpeza** – Atributos retirados após o merge e limpeza do dataset:

  - `teamID` no arquivo *teamStats.csv* e *games.csv*
  - `leagueID` em *appearances.csv* e *games.csv*
  - `playerID` em *shots.csv* e *appearances.csv*

    Todos esses atributos foram substituídos na limpeza do dataset (que ocorre no arquivo `merge_and_clean_dataset.py`) pelos nomes correspondentes presentes em outras tabelas, exemplo do jogador Philippe Coutinho que tinha id 488 em outras tabelas, o mesmo acontecia com times (teamID, homeTeamID e awayTeamID) e as ligas (leagueID). Todas essas colunas foram substituídas pelos seus nomes, e todos serão importantes para a clusterização, visto que o time em questão, a liga em que joga e o jogador em análise são informações relevantes e se correlacionam com outros dados (exemplo: Bundesliga tem muito mais gols que a Serie A).

    #### 🧹 Base de dados após a limpeza

    Após a limpeza e merge das tabelas escolhidas as tabelas se reduziaram a 4, pois as tabelas `players.csv`, `leagues.csv` e `teams.csv` tinham apenas 2 colunas: ID e nome, por isso os IDs forma substituidos pelos noems presentes nessas tabelas, restando 4 que foram renomeadas durante o resto da análise e cluesterização nos dataframes, que foram  `gameStats.csv`, `playerStatsinGame.csv`, `shotStats.csv` e `teamStats.csv`. A mudança de nome teve como objetivo uma descrição mais clara do dataframe, essas novas tabelas já tratadas forma armazenadas no diretório **dataset/trated_dataset**. 

    > 🔍 Esses atributos foram substituídos por seus respectivos nomes (ex: jogador, time, liga) para facilitar a interpretação durante a clusterização.

    - `appearances` → `playerStatsinGame`: <br>
    *playerID* → *playerName* <br>
    *leagueID* → *leagueName*

    - `teamStats` → `teamStats`: <br>
    *teamID** → *teamName* 

    - `shots` → `shotStats`: <br>
    *shooterID* → *shooterName* <br>
    *assisterID* → **assisterName*

    - `games` → `gameStats`: <br>
    *leagueID* → *leagueName* <br>
    *teamID* → *teamName*

    O atributo assisterName foi colocado no lugar de assisterID, porém, na coluna assisterID há muitos campos NA, pois, muitas finalizações (shots) podem não ter necessariamente um assistente, por isso NA, nesses casos foram mantidas NA.

    #### Date, season e gameID

    Outros atributos que não serão incluidos nessa análise inicial serão os atributos date e gameID, **date** presente nas tabelas `gameStats` e  `teamStats` e **gameID** presente nos 4 dataframes. Date por não ser uma informação relevante, tendo em vista que a análise não será tão profunda nesse momento, pois, a data da realização dos jogos (`gameStats`) e estatisticas dos times em um jogo (`teamStats`) provavelmente terá alguam relação quando se trata de inícios de temporada, finais de temporada ou pós a data-fifa (momento em que os clubes paralizam seus campeonatos para jogos entre seleções), entre outros. Para esse tipo de análise mais profunda, deixaremos em stand-by.

    Outro atributo que não será análisado ainda será o **season**, presente em `gameStats` e `teamStats`. Obviamente a informação em que temporada ocorreu o jogo pode impactar a análise de um jogador ou time, porém, para o tipo de análise inicial, não fará sentido a análise apenas desse atributo, pois ele varia igual para todos os times, o que faz sentido pois ele se trata da descrição da base de dados: dados coletados das temporadas 2015-2020 dos campeonatos nacionais das top 5 ligas europeias.

    Lembrando que em outras análises os atributos **season** e **date** serão levados em consideração devido a sua relevância em conjunto com outros dados entre as tabelas e métodos de análises mais profundas, porém, para a proposta inicial da tarefa, não será feita nesse momento. O atributo **gameID** foi retirado pois ele está na base dados apenas por uma questão de estruturação da base, já que foi feita em uma lógica de banco de dados relacionais, ele´e a chave estrangeira de muitas tabelas, portanto, o elo de ligação para relacionar dados entre jogos, jogadores e chutes.

<br>

---

<br>

2b. **Boxsplot dos atributos e verificação de outliers**

  A seguir estão os **boxplots gerados para análise de outliers e distribuição dos atributos** em cada conjunto de dados.  
  As imagens estão organizadas em grupos de 4 para melhor visualização.

  <p align="center">
  <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/B365A.png?raw=true" width="23%">
  <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/B365D.png?raw=true" width="23%">
  <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/B365H.png?raw=true" width="23%">
  <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/awayGoals.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/awayGoalsHalfTime.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/awayProbability.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/drawProbability.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/homeGoals.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/homeGoalsHalfTime.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/gameStats/homeProbability.png?raw=true" width="23%">
  </p>

  ---

  #### 👟 PlayerStatsInGame

  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/assists.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/goals.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/keyPasses.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/ownGoals.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/positionOrder.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/redCard.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/shots.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/substituteIn.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/substituteOut.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/time.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/xAssists.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/xGoals.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/xGoalsBuildUp.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/xGoalsChain.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/playerStatsinGame/yellowCard.png?raw=true" width="23%">
  </p>

  ---

  #### 🎯 ShotStats

  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/shotStats/minutes.png?raw=true" width="30%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/shotStats/positionX.png?raw=true" width="30%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/shotStats/positionY.png?raw=true" width="30%">
  </p>

  ---

  #### 🛡️ TeamStats

  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/corners.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/deep.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/fouls.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/goals.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/ppda.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/redCards.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/shots.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/shotsOnTarget.png?raw=true" width="23%">
  </p>
  <p align="center">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/xGoals.png?raw=true" width="23%">
    <img src="https://github.com/leonfagundes27/Assets/blob/main/boxsplot-datamining/teamStats/yellowCards.png?raw=true" width="23%">
  </p>

<br>

---

<br>

2c. **Transformação** – Aplicação de normalização, discretização e ajustes nos tipos de dados.

As transformações foram realizadas no script `data_transformation.py`, que aplica técnicas de **normalização e discretização** para facilitar a análise e clusterização dos dados. As alterações foram aplicadas diretamente nos dataframes tratados e salvos no diretório `dataset/transformed_dataset`.

### 🔁 Descrição das transformações aplicadas

#### ✅ Discretizações aplicadas

Os seguintes atributos numéricos foram discretizados em faixas categóricas, com os **rótulos em inglês** para manter o padrão internacional do dataset:

| Tabela                | Atributo original        | Faixas (bins)                          | Rótulos atribuídos                            |
|----------------------|--------------------------|----------------------------------------|------------------------------------------------|
| `gameStats`          | homeGoals                | 0, 1, 2+ gols                          | No Goals, Few Goals, Many Goals               |
| `gameStats`          | awayGoals                | 0, 1, 2+ gols                          | No Goals, Few Goals, Many Goals               |
| `gameStats`          | homeGoalsHalfTime        | 0, 1, 2+ gols                          | 0, 1, 2+                                       |
| `gameStats`          | awayGoalsHalfTime        | 0, 1, 2+ gols                          | 0, 1, 2+                                       |
| `playerStatsinGame`  | time (minutos jogados)   | 0–30, 31–60, 61–90                     | Low Time, Medium Time, High Time              |
| `shotStats`          | minute (minuto do chute) | Intervalos de 15 min                   | 0–15, 16–30, 31–45, 46–60, 61–75, 76–90        |
| `teamStats`          | ppda (pressão defensiva) | Alta, Média, Baixa (valores ajustados) | High Pressure, Medium Pressure, Low Pressure  |

#### 🔃 Normalizações com MinMaxScaler

Para garantir que os dados estejam na **mesma escala (0 a 1)** e evitar distorções na clusterização, foi aplicada a normalização Min-Max nos seguintes atributos numéricos:

| Tabela               | Atributos normalizados                                           |
|---------------------|------------------------------------------------------------------|
| `gameStats`         | homeProbability, drawProbability, awayProbability               |
| `playerStatsinGame` | xGoals, xGoalsChain, xGoalsBuildup, xAssists, shots             |
| `shotStats`         | xGoal, positionX, positionY                                     |
| `teamStats`         | shots, shotsOnTarget, goals, xGoals, deep                       |

Os arquivos resultantes dessas transformações foram salvos com o sufixo `_transformed.csv` no diretório `/dataset/transformed_dataset`.

> ✅ **Observação**: Todos os rótulos usados nas discretizações estão padronizados em **inglês**, mantendo a consistência do projeto e facilitando a aplicação de algoritmos de machine learning.


4. **Mineração de Dados (Clusterização)** – Definição do número ideal de clusters (método do cotovelo / *Elbow Method*) e análise dos agrupamentos.

5. **Avaliação e Interpretação dos Resultados** – Interpretação estatística e visual dos grupos formados.

</details>

---

### 🧮 Ferramentas e Tecnologias

- **Linguagem:** Python 🐍  
- **Bibliotecas:**  
  - `pandas`, `numpy` – manipulação de dados  
  - `matplotlib`, `seaborn` – visualização de gráficos  
  - `scikit-learn` – clusterização (KMeans, Elbow Method, etc.)  

---

### 🧑‍💻 Equipe de Desenvolvimento

- 👤 **[Erick Nascimento]**
- 👤 **[Juliano Ramalho]**
- 👤 **[Leon Fagundes]**

---

#### 📌 Licença

Este projeto é de uso **educacional e não comercial**.  
Desenvolvido para fins acadêmicos e de aprendizado em **Mineração de Dados e Machine Learning**.


