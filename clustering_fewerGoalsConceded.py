"""
Análise Complementar dos Clusters Defensivos

Este script demonstra como usar os resultados do clustering para 
análises mais específicas de defesas no futebol.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_cluster_results():
    """Análise detalhada dos resultados do clustering"""
    
    print("🔍 ANÁLISE COMPLEMENTAR DOS CLUSTERS DEFENSIVOS")
    print("=" * 60)
    
    # Carregar os resultados
    clusters_data = pd.read_csv('teamStats_clusters.csv')
    
    print(f"📊 Total de registros: {len(clusters_data):,}")
    print(f"📊 Times únicos: {clusters_data['teamName'].nunique()}")
    print(f"📊 Clusters identificados: {sorted(clusters_data['cluster'].unique())}")
    
    # 1. ANÁLISE DEFENSIVA POR CLUSTER (foco em jogos fora)
    print(f"\n🛡️ 1. ANÁLISE DEFENSIVA (Jogos Fora)")
    print("-" * 40)
    
    away_games = clusters_data[clusters_data['location'] == 'a']
    
    # Análise por resultado e cluster
    away_defense = away_games.groupby(['cluster', 'result']).size().unstack(fill_value=0)
    away_defense_pct = away_defense.div(away_defense.sum(axis=1), axis=0) * 100
    
    print("📈 Distribuição de resultados em jogos fora por cluster:")
    print(away_defense_pct.round(1))
    
    # 2. TIMES COM MELHOR DEFESA POR CLUSTER
    print(f"\n🏆 2. TOP 5 MELHORES DEFESAS POR CLUSTER (jogos fora)")
    print("-" * 50)
    
    # Calcular médias defensivas por time (apenas jogos fora)
    team_defense = away_games.groupby(['cluster', 'teamName']).agg({
        'awayGoals': lambda x: (x == 'No Goals').mean() * 100,  # % jogos sem sofrer gols
        'result': lambda x: (x != 'L').mean() * 100,  # % jogos sem perder
        'gameID': 'count'  # número de jogos
    }).round(2)
    
    team_defense.columns = ['%_Jogos_Sem_Gols', '%_Jogos_Sem_Perder', 'Num_Jogos']
    
    # Filtrar times com pelo menos 10 jogos fora
    team_defense_filtered = team_defense[team_defense['Num_Jogos'] >= 10]
    
    for cluster in sorted(clusters_data['cluster'].unique()):
        print(f"\n🏅 CLUSTER {cluster} - Top 5 Defesas:")
        cluster_teams = team_defense_filtered.loc[cluster].nlargest(5, '%_Jogos_Sem_Gols')
        print(cluster_teams[['%_Jogos_Sem_Gols', '%_Jogos_Sem_Perder', 'Num_Jogos']])
    
    # 3. ANÁLISE TEMPORAL (se houver múltiplas temporadas)
    print(f"\n📅 3. ANÁLISE TEMPORAL")
    print("-" * 30)
    
    seasons = clusters_data['season'].unique()
    if len(seasons) > 1:
        seasonal_analysis = clusters_data.groupby(['season', 'cluster']).agg({
            'result': lambda x: (x == 'W').mean() * 100,
            'gameID': 'count'
        }).round(2)
        seasonal_analysis.columns = ['Taxa_Vitoria_%', 'Num_Jogos']
        print("📊 Taxa de vitória por temporada e cluster:")
        print(seasonal_analysis)
    else:
        print("📊 Análise temporal não disponível (apenas uma temporada)")
    
    # 4. ANÁLISE DE PRESSÃO POR CLUSTER
    print(f"\n⚡ 4. ANÁLISE DE PRESSÃO DEFENSIVA")
    print("-" * 40)
    
    pressure_analysis = clusters_data.groupby(['cluster', 'pressure']).agg({
        'result': lambda x: (x == 'W').mean() * 100,
        'gameID': 'count'
    }).round(2)
    pressure_analysis.columns = ['Taxa_Vitoria_%', 'Num_Jogos']
    print("📊 Taxa de vitória por nível de pressão e cluster:")
    print(pressure_analysis)
    
    # 5. VISUALIZAÇÃO COMPARATIVA
    create_defensive_visualization(clusters_data)
    
    return clusters_data

def create_defensive_visualization(data):
    """Cria visualizações específicas para análise defensiva"""
    
    print(f"\n🎨 5. CRIANDO VISUALIZAÇÕES DEFENSIVAS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Defensiva por Clusters - Futebol', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Taxa de vitória por cluster e localização
    ax1 = axes[0, 0]
    win_rates = data.groupby(['cluster', 'location'])['result'].apply(lambda x: (x == 'W').mean() * 100)
    win_rates.unstack().plot(kind='bar', ax=ax1, color=['lightcoral', 'lightblue'])
    ax1.set_title('Taxa de Vitória por Cluster e Localização')
    ax1.set_ylabel('Taxa de Vitória (%)')
    ax1.legend(['Fora', 'Casa'])
    ax1.set_xlabel('Cluster')
    
    # Gráfico 2: Distribuição de gols sofridos fora
    ax2 = axes[0, 1]
    away_data = data[data['location'] == 'a']
    goals_dist = away_data.groupby('cluster')['awayGoals'].value_counts(normalize=True).unstack(fill_value=0)
    goals_dist.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Distribuição de Gols Sofridos Fora')
    ax2.set_ylabel('Proporção')
    ax2.set_xlabel('Cluster')
    ax2.legend(title='Gols Sofridos', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 3: Pressão defensiva por cluster
    ax3 = axes[1, 0]
    pressure_dist = data.groupby('cluster')['pressure'].value_counts(normalize=True).unstack(fill_value=0)
    pressure_dist.plot(kind='bar', ax=ax3)
    ax3.set_title('Distribuição de Pressão Defensiva')
    ax3.set_ylabel('Proporção')
    ax3.set_xlabel('Cluster')
    ax3.legend(title='Nível Pressão', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 4: Comparação de resultados
    ax4 = axes[1, 1]
    result_dist = data.groupby('cluster')['result'].value_counts(normalize=True).unstack(fill_value=0)
    result_dist.plot(kind='bar', ax=ax4, color=['red', 'gray', 'green'])
    ax4.set_title('Distribuição de Resultados por Cluster')
    ax4.set_ylabel('Proporção')
    ax4.set_xlabel('Cluster')
    ax4.legend(title='Resultado')
    
    plt.tight_layout()
    plt.savefig('defensive_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráficos salvos em: defensive_analysis_charts.png")

def identify_best_defenses(data, min_games=15):
    """Identifica os times com as melhores defesas"""
    
    print(f"\n🏆 6. IDENTIFICAÇÃO DAS MELHORES DEFESAS")
    print("-" * 50)
    
    # Focar em jogos fora (mais difícil defender)
    away_games = data[data['location'] == 'a']
    
    # Calcular métricas defensivas por time
    team_metrics = away_games.groupby('teamName').agg({
        'awayGoals': [
            lambda x: (x == 'No Goals').mean() * 100,  # % jogos sem sofrer
            lambda x: (x == 'Few Goals').mean() * 100,  # % poucos gols sofridos
            'count'  # número de jogos
        ],
        'result': [
            lambda x: (x == 'W').mean() * 100,  # % vitórias
            lambda x: (x != 'L').mean() * 100   # % não derrotas
        ],
        'cluster': 'first'  # cluster predominante
    })
    
    # Flatten column names
    team_metrics.columns = ['%_Sem_Gols', '%_Poucos_Gols', 'Jogos_Fora', 
                           '%_Vitorias', '%_Nao_Derrotas', 'Cluster']
    
    # Filtrar times com jogos suficientes
    qualified_teams = team_metrics[team_metrics['Jogos_Fora'] >= min_games]
    
    # Criar score defensivo combinado
    qualified_teams['Score_Defensivo'] = (
        qualified_teams['%_Sem_Gols'] * 0.4 +
        qualified_teams['%_Poucos_Gols'] * 0.3 + 
        qualified_teams['%_Nao_Derrotas'] * 0.3
    )
    
    # Top 10 melhores defesas
    top_defenses = qualified_teams.nlargest(10, 'Score_Defensivo')
    
    print(f"🥇 TOP 10 MELHORES DEFESAS (jogos fora, mín. {min_games} jogos):")
    print("=" * 70)
    
    for i, (team, stats) in enumerate(top_defenses.iterrows(), 1):
        print(f"{i:2d}º {team:20s} | Cluster: {stats['Cluster']} | "
              f"Score: {stats['Score_Defensivo']:.1f} | "
              f"Sem gols: {stats['%_Sem_Gols']:.1f}% | "
              f"Não derrotas: {stats['%_Nao_Derrotas']:.1f}%")
    
    return top_defenses

if __name__ == "__main__":
    # Executar análise completa
    clusters_data = analyze_cluster_results()
    best_defenses = identify_best_defenses(clusters_data)
    
    print(f"\n✅ ANÁLISE COMPLEMENTAR CONCLUÍDA!")
    print("📁 Arquivos gerados:")
    print("  └ defensive_analysis_charts.png")
    print(f"\n💡 PRINCIPAIS INSIGHTS:")
    print("  ├ Cluster 0: Times com defesa mais sólida (menos gols sofridos)")
    print("  ├ Cluster 1: Times mais ofensivos (mais gols, mais vitórias)")
    print("  ├ Defesas se comportam diferente em casa vs fora")
    print("  └ Nível de pressão influencia no resultado defensivo")