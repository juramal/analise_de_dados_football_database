# ⚽ Football Database

Bem-vindo ao **Football Database**, um projeto acadêmico voltado à análise e mineração de dados de partidas de futebol utilizando técnicas de **KDD (Knowledge Discovery in Databases)** e **Clusterização com Python**.

Este projeto tem como objetivo compreender padrões de desempenho de jogadores, times e ligas, a partir de múltiplos datasets interligados.

---

## 🧠 Tarefas do Projeto

<details>
<summary><b>🔗 Ver TAREFA 1 – Processo KDD e Clusterização</b></summary>

## 📂 Estrutura do Projeto

| Diretório / Arquivo | Descrição |
|----------------------|------------|
| `/datasets/` | Contém os arquivos CSV originais (appearances, games, players, teams, etc.) |
| `/scripts/` | Códigos Python utilizados para limpeza, transformação e clusterização dos dados |
| `/notebooks/` | Notebooks Jupyter usados para visualização, análise e geração dos gráficos |
| `/images/` | Imagens geradas durante o processo de análise (boxplots, elbow method, clusters, etc.) |
| `README.md` | Este arquivo explicativo do projeto |

---

## 🧩 Etapas do Processo KDD

O projeto segue as seguintes etapas do processo de **KDD**:

1. **Seleção de Dados** – Escolha dos datasets relevantes:  
   - `appearances.csv`  
   - `games.csv`  
   - `players.csv`  
   - `shots.csv`  
   - `teams.csv`  
   - `leagues.csv`  
   - `teamstats.csv`

2. **Pré-processamento e Limpeza** – Exclusão de campos desnecessários e substituição de IDs por nomes correspondentes.  
   > Exemplo: substituição de `playerID`, `teamID`, `leagueID` por nomes reais dos jogadores, times e ligas.  

3. **Transformação** – Aplicação de normalização, discretização e ajustes nos tipos de dados.

4. **Mineração de Dados (Clusterização)** – Definição do número ideal de clusters (método do cotovelo / *Elbow Method*) e análise dos agrupamentos.

5. **Avaliação e Interpretação dos Resultados** – Interpretação estatística e visual dos grupos formados.

---

## 📊 Dados Utilizados

Cada dataset contém atributos relevantes para a análise.  
Exemplo do dataset `appearances`:

### 🧾 Dataset: `appearances`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| gameID | int | Identificador do jogo |
| payerID | str | Identificador do jogador |
| goals | int | Número de gols marcados |
| ownGoals | int | Número de gols contra |
| shots | int | Número de chutes do jogador |
| xGoals | double | Probabilidade de um chute resultar em gol |
| xGoalsChain | double | Posse de bola que resultou em um chute |
| xGoalsBuildup | double | Contribuição do jogador para um chute |
| assists | int | Número de assistências |
| keyPasses | int | Passe final antes de um chute |
| xAssists | double | Probabilidade de uma assistência resultar em gol |
| position | str | Posição do jogador em campo |
| positionOrder | int | Ordem da posição em campo |
| yellowCard | int | Cartões amarelos recebidos |
| redCard | int | Cartões vermelhos recebidos |
| time | int | Minutos jogados |
| substituteIn | int | Jogador entrou em campo |
| substituteOut | int | Jogador saiu de campo |
| leagueID | int | Identificador da liga |

---

### 🏟️ Dataset: `games`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| gameID | int | Identificador do jogo |
| leagueID | int | Identificador da liga |
| season | int | Ano da temporada |
| date | date_time | Data e hora do jogo |
| homeTeamID | int | Identificador do time da casa |
| awayTeamID | int | Identificador do time visitante |
| homeGoals | int | Gols do time da casa |
| awayGoals | int | Gols do time visitante |
| homeProbability | float | Probabilidade de vitória do time da casa |
| drawProbability | float | Probabilidade de empate |
| awayProbability | float | Probabilidade de vitória do time visitante |
| homeGoalsHalfTime | int | Gols do time da casa no intervalo |
| awayGoalsHalfTime | int | Gols do time visitante no intervalo |
| B365H | float | Bet365: vitória time da casa |
| B365D | float | Bet365: empate |
| B365A | float | Bet365: vitória time visitante |
| BWH | float | BW: vitória time da casa |
| BWD | float | BW: empate |
| BWA | float | BW: vitória time visitante |
| IWH | float | IW: vitória time da casa |
| IWD | float | IW: empate |
| IWA | float | IW: vitória time visitante |
| PSH | float | PS: vitória time da casa |
| PSD | float | PS: empate |
| PSA | float | PS: vitória time visitante |
| WHH | float | WH: vitória time da casa |
| WHD | float | WH: empate |
| WHA | float | WH: vitória time visitante |
| VCH | float | VC: vitória time da casa |
| VCD | float | VC: empate |
| VCA | float | VC: vitória time visitante |

---

### 🏆 Dataset: `leagues`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| leagueID | int | Identificador da liga |
| name | str | Nome da liga |
| understatNotation | str | Sigla de identificação da liga |

---

### 👟 Dataset: `players`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| playerID | int | Identificador do jogador |
| name | str | Nome do jogador |

---

### 🎯 Dataset: `shots`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| gameID | int | Identificador do jogo |
| shooterID | int | Jogador que realizou o chute |
| assisterID | int | Jogador que deu assistência |
| minute | int | Minuto do jogo do chute |
| situation | str | Tipo de lance (ex: bola parada, contra-ataque) |
| lastAction | str | Tipo da jogada anterior |
| shotType | str | Pé utilizado no chute (esquerdo/direito) |
| shotResult | str | Resultado do chute (gol, fora, bloqueado, etc.) |
| xGoal | double | Probabilidade de um chute resultar em gol |
| positionX | double | Coordenada X do chute |
| positionY | double | Coordenada Y do chute |

---

### 🛡️ Dataset: `teams`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| teamID | int | Identificador do time |
| name | str | Nome do time |

---

### 📈 Dataset: `teamstats`

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| gameID | int | Identificador do jogo |
| teamID | int | Identificador do time |
| season | int | Ano da temporada |
| date | date_time | Data e hora do jogo |
| location | str | “h” para time da casa, “a” para visitante |
| goals | int | Gols marcados pelo time |
| xGoals | double | Probabilidade de um chute resultar em gol |
| shots | int | Total de chutes realizados |
| shotsOnTarget | int | Chutes no gol |
| deep | int | Lances de fundo de área |
| ppda | double | Índice de retomada de bola |
| fouls | int | Faltas cometidas |
| corners | int | Escanteios recebidos |
| yellowCards | int | Cartões amarelos |
| redCards | int | Cartões vermelhos |
| result | str | Resultado do jogo (“W”, “L”, “D”) |

---

Outros datasets incluem informações complementares sobre jogos, times, estatísticas e ligas.

---

## 🧮 Ferramentas e Tecnologias

- **Linguagem:** Python 🐍  
- **Bibliotecas:**  
  - `pandas`, `numpy` – manipulação de dados  
  - `matplotlib`, `seaborn` – visualização de gráficos  
  - `scikit-learn` – clusterização (KMeans, Elbow Method, etc.)  
- **Ambiente:** Jupyter Notebook / VSCode

--- 



<br>

Nesta tarefa serão incluídas **as informações detalhadas do processo KDD**, **as imagens dos gráficos** e **as análises dos clusters** conforme o desenvolvimento avança.


---

### 📋 Descrição Geral
O objetivo desta tarefa é aplicar o processo de **descoberta de conhecimento em bases de dados (KDD)** no conjunto de dados de futebol, realizando as etapas de:
1. Seleção dos datasets relevantes  
2. Limpeza e integração dos dados  
3. Normalização e transformação  
4. Execução da **clusterização (K-Means)**  
5. Interpretação dos resultados obtidos

---

### 🧩 Datasets Utilizados
- `appearances.csv`
- `games.csv`
- `players.csv`
- `teams.csv`
- `leagues.csv`
- `shots.csv`
- `teamstats.csv`

---

### ⚙️ Pré-processamento
Atributos removidos após o merge:
- `teamID` em *teamStats.csv* e *games.csv*  
- `leagueID` em *appearances.csv* e *games.csv*  
- `playerID` em *shots.csv* e *appearances.csv*

> 🔍 Esses atributos foram substituídos por seus respectivos nomes (ex: jogador, time, liga) para facilitar a interpretação durante a clusterização.

---

### 📊 Boxplots por Categoria de Dados

A seguir estão os **boxplots gerados para análise de outliers e distribuição dos atributos** em cada conjunto de dados.  
As imagens estão organizadas em grupos de 4 para melhor visualização.

---

#### 🏟️ GameStats

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


---

### 🧠 Análise dos Resultados
As interpretações e conclusões obtidas com base nos clusters formados serão descritas aqui conforme o desenvolvimento avança.

> Exemplo: “O Cluster 1 representa jogadores com alta taxa de gols e participação ofensiva, enquanto o Cluster 2 agrupa defensores com alta eficiência em desarmes.”

<summary><b>🔗 FIM da Tarefa 1 – Processo KDD e Clusterização</b></summary>
</details>
---

## 🧑‍💻 Equipe de Desenvolvimento

- 👤 **[Erick Nascimento]**
- 👤 **[Juliano Ramalho]**
- 👤 **[Leon Fagundes]**

---



## 📌 Licença

Este projeto é de uso **educacional e não comercial**.  
Desenvolvido para fins acadêmicos e de aprendizado em **Mineração de Dados e Machine Learning**.

---

### ⭐ Dica:
Se este projeto te ajudou, **deixe uma estrela** no repositório para apoiar o desenvolvimento! 🌟

---

