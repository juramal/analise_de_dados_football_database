
    # Como usar o arquivo teamStats_clusters.csv
    
    Este arquivo contém os resultados da análise de clustering para defesas de futebol.
    
    ## Colunas principais:
    - gameID: Identificador único do jogo
    - teamName: Nome do time
    - season: Temporada
    - date: Data do jogo  
    - location: Localização (h=casa, a=fora)
    - cluster: Cluster atribuído (0, 1, 2, ...)
    - pressure: Nível de pressão do time
    - result: Resultado (W=vitória, L=derrota, D=empate)
    - awayGoals/homeGoals: Gols marcados fora/casa
    
    ## Análises sugeridas:
    1. Comparar performance defensiva entre clusters
    2. Analisar padrões de jogos em casa vs fora por cluster
    3. Correlacionar clusters com outras métricas (xGoals, shots, etc.)
    4. Identificar times que mudaram de cluster ao longo das temporadas
    
    ## Exemplo de uso:
    ```python
    import pandas as pd
    data = pd.read_csv('teamStats_clusters.csv')
    
    # Análise de defesa por cluster (jogos fora)
    defense_analysis = data[data['location'] == 'a'].groupby('cluster')['goals'].mean()
    print("Gols sofridos fora por cluster:", defense_analysis.sort_values())
    ```
    