# ⚽ Football Database

Bem-vindo ao **Football Database**, um projeto acadêmico voltado à análise e mineração de dados de partidas de futebol utilizando técnicas de **KDD (Knowledge Discovery in Databases)** e **Clusterização com Python**.

Este projeto tem como objetivo compreender padrões de desempenho de jogadores, times e ligas, a partir de múltiplos datasets interligados.

---

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

| Atributo | Tipo | Descrição |
|-----------|------|------------|
| gameID | int | Identificador do jogo |
| playerID | str | Identificador do jogador |
| goals | int | Número de gols |
| assists | int | Número de assistências |
| position | str | Posição do jogador em campo |
| yellowCard | int | Cartões amarelos recebidos |
| redCard | int | Cartões vermelhos recebidos |
| time | int | Minutos jogados |

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

## 🧠 Tarefas do Projeto

### 🔗 [Tarefa 1 – Processo KDD e Clusterização](./tarefa1.md)
> Clique acima para abrir a primeira tarefa.  
> Neste arquivo serão inseridas **as informações detalhadas do processo**, **as imagens dos gráficos** e **as análises dos clusters** conforme o desenvolvimento avança.

---




## 🧑‍💻 Equipe de Desenvolvimento

- 👤 **[Erick Nascimento]**
- 👤 **[Juliano Ramalho]**
- 👤 **[Leon Fagundes]**

---



## 📸 Imagens e Resultados

As imagens (boxplots, gráficos de Elbow e clusters) serão adicionadas conforme forem geradas.

> 📷 **Exemplo de prévia (em breve):**
>
> ![Boxplot exemplo](./images/boxplot_exemplo.png)
>
> ![Elbow Method exemplo](./images/elbow_exemplo.png)

---

## 📌 Licença

Este projeto é de uso **educacional e não comercial**.  
Desenvolvido para fins acadêmicos e de aprendizado em **Mineração de Dados e Machine Learning**.

---

### ⭐ Dica:
Se este projeto te ajudou, **deixe uma estrela** no repositório para apoiar o desenvolvimento! 🌟

---

