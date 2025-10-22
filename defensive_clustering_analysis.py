"""
Script de Clustering para Análise de Defesas no Futebol

Este script realiza uma análise completa de clustering para identificar padrões
defensivos em times de futebol, com foco especial na variável 'awayGoals' para
encontrar os times com as melhores defesas.

Autor: Assistente de Ciência de Dados
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def calculate_dunn_index(X, labels):
    """
    Calcula o Índice de Dunn para avaliação de clustering.
    Dunn Index = min(distância entre clusters) / max(diâmetro intra-cluster)
    Valores maiores indicam melhor clustering.
    """
    from scipy.spatial.distance import pdist, cdist
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # Calcular distâncias mínimas entre clusters
    min_inter_cluster_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            
            # Distância mínima entre pontos dos dois clusters
            distances = cdist(cluster_i, cluster_j)
            min_dist = np.min(distances)
            min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)
    
    # Calcular diâmetro máximo intra-cluster
    max_intra_cluster_dist = 0.0
    for i in range(n_clusters):
        cluster_points = X[labels == unique_labels[i]]
        if len(cluster_points) > 1:
            # Diâmetro do cluster (máxima distância entre pontos do cluster)
            cluster_distances = pdist(cluster_points)
            if len(cluster_distances) > 0:
                max_cluster_dist = np.max(cluster_distances)
                max_intra_cluster_dist = max(max_intra_cluster_dist, max_cluster_dist)
    
    # Calcular índice de Dunn
    if max_intra_cluster_dist > 0:
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
    else:
        dunn_index = 0.0
    
    return dunn_index

def load_and_merge_data():
    """
    Carrega e faz merge dos datasets gameStats e teamStats.
    
    Returns:
        pd.DataFrame: Dataset mesclado e limpo
    """
    print("🔄 Carregando datasets...")
    
    # Carregar datasets
    team_stats = pd.read_csv('dataset/transformed_dataset/teamStats_transformed.csv')
    game_stats = pd.read_csv('dataset/transformed_dataset/gameStats_transformed.csv')
    
    print(f"📊 TeamStats: {team_stats.shape[0]} registros, {team_stats.shape[1]} colunas")
    print(f"📊 GameStats: {game_stats.shape[0]} registros, {game_stats.shape[1]} colunas")
    
    # Garantir que gameID seja numérico
    team_stats['gameID'] = pd.to_numeric(team_stats['gameID'], errors='coerce')
    game_stats['gameID'] = pd.to_numeric(game_stats['gameID'], errors='coerce')
    
    # Remover registros com gameID inválido
    team_stats = team_stats.dropna(subset=['gameID'])
    game_stats = game_stats.dropna(subset=['gameID'])
    
    # Fazer merge das tabelas
    merged_data = team_stats.merge(game_stats, on='gameID', how='inner', suffixes=('', '_game'))
    
    print(f"📊 Dataset mesclado: {merged_data.shape[0]} registros, {merged_data.shape[1]} colunas")
    
    # Remover duplicatas
    initial_rows = merged_data.shape[0]
    merged_data = merged_data.drop_duplicates()
    print(f"🧹 Removidas {initial_rows - merged_data.shape[0]} duplicatas")
    
    # Remover colunas redundantes (sufixo _game que já existem)
    columns_to_drop = ['season_game', 'date_game']
    existing_cols_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
    merged_data = merged_data.drop(columns=existing_cols_to_drop)
    
    return merged_data

def prepare_features(data):
    """
    Prepara as features para clustering com foco em análise defensiva.
    
    Args:
        data (pd.DataFrame): Dataset mesclado
        
    Returns:
        tuple: (features_scaled, scaler, feature_names, processed_data)
    """
    print("🔧 Preparando features para análise defensiva...")
    
    # Criar variável binária para localização
    data['location_bin'] = (data['location'] == 'h').astype(int)
    
    # Selecionar features para análise defensiva
    defensive_features = [
        # Features defensivas principais
        'goals',  # Gols sofridos/marcados
        'xGoals',  # Expected goals
        'shots',  # Chutes permitidos/realizados
        'shotsOnTarget',  # Chutes no gol
        'deep',  # Ataques profundos
        'ppda',  # Passes por ação defensiva (importante para defesa)
        'fouls',  # Faltas cometidas
        'corners',  # Escanteios
        'yellowCards',  # Cartões amarelos
        'redCards',  # Cartões vermelhos
        
        # Features do jogo (probabilidades e odds)
        'homeProbability',
        'drawProbability', 
        'awayProbability',
        'B365H', 'B365D', 'B365A',  # Odds Bet365
        'location_bin'  # Home/Away
    ]
    
    # Verificar quais features existem no dataset
    available_features = [col for col in defensive_features if col in data.columns]
    missing_features = [col for col in defensive_features if col not in data.columns]
    
    if missing_features:
        print(f"⚠️ Features não encontradas: {missing_features}")
    
    print(f"✅ Features selecionadas para clustering: {len(available_features)}")
    print(f"📋 Features: {available_features}")
    
    # Criar dataset apenas com features numéricas
    features_data = data[available_features].copy()
    
    # Tratar valores infinitos e ausentes
    print("🔧 Tratando valores infinitos e ausentes...")
    features_data = features_data.replace([np.inf, -np.inf], np.nan)
    
    # Preencher valores ausentes com a mediana
    for col in features_data.columns:
        if features_data[col].isnull().sum() > 0:
            median_val = features_data[col].median()
            features_data[col].fillna(median_val, inplace=True)
            print(f"  └ {col}: {features_data[col].isnull().sum()} valores preenchidos com mediana ({median_val:.3f})")
    
    # Aplicar StandardScaler para normalização
    print("🔧 Aplicando normalização StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_data)
    
    # Adicionar informações importantes para análise posterior
    processed_data = data[['gameID', 'teamName', 'season', 'date', 'location', 
                          'result', 'pressure', 'awayGoals', 'homeGoals', 'goals']].copy()
    
    return features_scaled, scaler, available_features, processed_data

def find_optimal_clusters(features_scaled, max_clusters=10):
    """
    Encontra o número ótimo de clusters usando múltiplas métricas de avaliação.
    
    Args:
        features_scaled (np.array): Features normalizadas
        max_clusters (int): Número máximo de clusters para testar
        
    Returns:
        int: Número ótimo de clusters
    """
    print("📈 Analisando número ótimo de clusters com múltiplas métricas...")
    
    # Listas para armazenar métricas
    k_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    dunn_indices = []
    
    for k in k_range:
        print(f"  └ Testando k={k}...")
        
        # Treinar KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        inertias.append(kmeans.inertia_)
        
        sil_score = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        
        db_score = davies_bouldin_score(features_scaled, cluster_labels)
        davies_bouldin_scores.append(db_score)
        
        dunn_score = calculate_dunn_index(features_scaled, cluster_labels)
        dunn_indices.append(dunn_score)
        
        print(f"    ├ Inertia: {kmeans.inertia_:.2f}")
        print(f"    ├ Silhouette: {sil_score:.3f}")
        print(f"    ├ Davies-Bouldin: {db_score:.3f}")
        print(f"    └ Dunn Index: {dunn_score:.3f}")
    
    # Encontrar k ótimo baseado no Silhouette Score (critério principal)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    # Encontrar outros ótimos para comparação
    min_db_k = k_range[np.argmin(davies_bouldin_scores)]  # Menor Davies-Bouldin é melhor
    max_dunn_k = k_range[np.argmax(dunn_indices)]  # Maior Dunn é melhor
    
    print(f"✅ Análise de métricas:")
    print(f"  ├ Melhor Silhouette: k={optimal_k} (Score: {max_silhouette:.3f})")
    print(f"  ├ Melhor Davies-Bouldin: k={min_db_k} (Score: {min(davies_bouldin_scores):.3f})")
    print(f"  └ Melhor Dunn Index: k={max_dunn_k} (Score: {max(dunn_indices):.3f})")
    
    # Plotar gráficos de análise com as 4 métricas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise de Métricas para Determinação do Número Ótimo de Clusters', 
                 fontsize=16, fontweight='bold')
    
    # Gráfico 1: Método do cotovelo (Inertia)
    ax1 = axes[0, 0]
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Soma dos Erros Quadráticos (Inertia)', fontsize=12)
    ax1.set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_range)
    
    # Gráfico 2: Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=max_silhouette, color='g', linestyle='--', alpha=0.7, 
                label=f'Máximo: k={optimal_k} (Score={max_silhouette:.3f})')
    ax2.axvline(x=optimal_k, color='g', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score (Maior = Melhor)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(k_range)
    
    # Gráfico 3: Davies-Bouldin Score
    ax3 = axes[1, 0]
    ax3.plot(k_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
    min_db_score = min(davies_bouldin_scores)
    ax3.axhline(y=min_db_score, color='r', linestyle='--', alpha=0.7,
                label=f'Mínimo: k={min_db_k} (Score={min_db_score:.3f})')
    ax3.axvline(x=min_db_k, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax3.set_ylabel('Davies-Bouldin Score', fontsize=12)
    ax3.set_title('Índice Davies-Bouldin (Menor = Melhor)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticks(k_range)
    
    # Gráfico 4: Dunn Index
    ax4 = axes[1, 1]
    ax4.plot(k_range, dunn_indices, 'mo-', linewidth=2, markersize=8)
    max_dunn_score = max(dunn_indices)
    ax4.axhline(y=max_dunn_score, color='orange', linestyle='--', alpha=0.7,
                label=f'Máximo: k={max_dunn_k} (Score={max_dunn_score:.3f})')
    ax4.axvline(x=max_dunn_k, color='orange', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax4.set_ylabel('Dunn Index', fontsize=12)
    ax4.set_title('Índice Dunn (Maior = Melhor)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xticks(k_range)
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Criar gráfico comparativo das métricas normalizadas
    create_normalized_metrics_comparison(k_range, silhouette_scores, davies_bouldin_scores, 
                                       dunn_indices, optimal_k, min_db_k, max_dunn_k)
    
    return optimal_k

def create_normalized_metrics_comparison(k_range, silhouette_scores, davies_bouldin_scores, 
                                       dunn_indices, optimal_k, min_db_k, max_dunn_k):
    """
    Cria gráfico comparativo com métricas normalizadas.
    """
    # Normalizar métricas para comparação (0-1)
    def normalize_metric(values, higher_better=True):
        values = np.array(values)
        if higher_better:
            # Para métricas onde maior é melhor (Silhouette, Dunn)
            return (values - values.min()) / (values.max() - values.min())
        else:
            # Para métricas onde menor é melhor (Davies-Bouldin) - inverter
            return 1 - (values - values.min()) / (values.max() - values.min())
    
    sil_norm = normalize_metric(silhouette_scores, True)
    db_norm = normalize_metric(davies_bouldin_scores, False)  # Inverter: menor é melhor
    dunn_norm = normalize_metric(dunn_indices, True)
    
    plt.figure(figsize=(12, 8))
    
    # Plotar métricas normalizadas
    plt.plot(k_range, sil_norm, 'ro-', linewidth=2, markersize=8, 
             label='Silhouette Score (normalizado)', alpha=0.8)
    plt.plot(k_range, db_norm, 'go-', linewidth=2, markersize=8, 
             label='Davies-Bouldin (inverso normalizado)', alpha=0.8)
    plt.plot(k_range, dunn_norm, 'mo-', linewidth=2, markersize=8, 
             label='Dunn Index (normalizado)', alpha=0.8)
    
    # Marcar pontos ótimos
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Ótimo Silhouette (k={optimal_k})')
    plt.axvline(x=min_db_k, color='green', linestyle='--', alpha=0.7, 
                label=f'Ótimo Davies-Bouldin (k={min_db_k})')
    plt.axvline(x=max_dunn_k, color='magenta', linestyle='--', alpha=0.7, 
                label=f'Ótimo Dunn (k={max_dunn_k})')
    
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Valor Normalizado (0-1)', fontsize=12)
    plt.title('Comparação de Métricas de Avaliação de Clustering\n(Normalizadas para Facilitar Comparação)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(k_range)
    plt.ylim(0, 1.1)
    
    # Adicionar anotações com valores
    for i, k in enumerate(k_range):
        plt.annotate(f'{sil_norm[i]:.2f}', (k, sil_norm[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('normalized_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Gráficos salvos:")
    print("  ├ cluster_evaluation_metrics.png (métricas individuais)")
    print("  └ normalized_metrics_comparison.png (comparação normalizada)")

def train_kmeans(features_scaled, n_clusters):
    """
    Treina o modelo KMeans final com o número ótimo de clusters.
    
    Args:
        features_scaled (np.array): Features normalizadas
        n_clusters (int): Número de clusters
        
    Returns:
        tuple: (modelo_kmeans, labels_clusters)
    """
    print(f"🎯 Treinando modelo KMeans final com {n_clusters} clusters...")
    
    # Treinar modelo final
    kmeans_final = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=30,  # Múltiplas inicializações para melhor convergência
        max_iter=300
    )
    
    cluster_labels = kmeans_final.fit_predict(features_scaled)
    
    # Calcular métricas finais
    final_inertia = kmeans_final.inertia_
    final_silhouette = silhouette_score(features_scaled, cluster_labels)
    
    print(f"✅ Modelo treinado com sucesso!")
    print(f"  ├ Inertia final: {final_inertia:.2f}")
    print(f"  └ Silhouette Score final: {final_silhouette:.3f}")
    
    return kmeans_final, cluster_labels

def analyze_clusters(processed_data, cluster_labels, features_scaled, scaler, feature_names):
    """
    Analisa e interpreta os clusters formados com foco na análise defensiva.
    
    Args:
        processed_data (pd.DataFrame): Dados processados
        cluster_labels (np.array): Labels dos clusters
        features_scaled (np.array): Features normalizadas
        scaler (StandardScaler): Scaler usado na normalização
        feature_names (list): Nomes das features
    """
    print("🔍 Analisando e interpretando clusters...")
    
    # Adicionar labels de cluster aos dados
    processed_data['cluster'] = cluster_labels
    
    # Reverter normalização para interpretação (centroides originais)
    features_original = scaler.inverse_transform(features_scaled)
    features_df = pd.DataFrame(features_original, columns=feature_names)
    features_df['cluster'] = cluster_labels
    
    n_clusters = len(np.unique(cluster_labels))
    
    print(f"\n📊 ANÁLISE DOS {n_clusters} CLUSTERS IDENTIFICADOS")
    print("=" * 80)
    
    # Análise dos centroides por cluster
    for cluster in range(n_clusters):
        cluster_data = processed_data[processed_data['cluster'] == cluster]
        cluster_features = features_df[features_df['cluster'] == cluster]
        
        print(f"\n🏆 CLUSTER {cluster} - {len(cluster_data)} jogos")
        print("-" * 50)
        
        # Estatísticas das features principais para defesa
        main_defensive_features = ['goals', 'xGoals', 'shots', 'shotsOnTarget', 'ppda', 'fouls']
        available_def_features = [f for f in main_defensive_features if f in feature_names]
        
        print("📈 Características Defensivas (médias):")
        for feature in available_def_features:
            if feature in cluster_features.columns:
                mean_val = cluster_features[feature].mean()
                std_val = cluster_features[feature].std()
                print(f"  ├ {feature}: {mean_val:.3f} (±{std_val:.3f})")
        
        # Análise específica de gols sofridos quando jogando fora (defesa)
        away_games = cluster_data[cluster_data['location'] == 'a']
        if len(away_games) > 0:
            avg_goals_conceded_away = away_games['goals'].mean()  # Gols sofridos fora
            print(f"  ├ Gols sofridos fora (média): {avg_goals_conceded_away:.3f}")
        
        # Análise de resultados
        result_dist = cluster_data['result'].value_counts(normalize=True) * 100
        print("📊 Distribuição de Resultados:")
        for result, pct in result_dist.items():
            print(f"  ├ {result}: {pct:.1f}%")
        
        # Taxa de vitória
        win_rate = (cluster_data['result'] == 'W').mean() * 100
        print(f"🏅 Taxa de Vitória: {win_rate:.1f}%")
        
        # Análise de pressão
        if 'pressure' in cluster_data.columns:
            pressure_dist = cluster_data['pressure'].value_counts(normalize=True) * 100
            print("⚡ Distribuição de Pressão:")
            for pressure, pct in pressure_dist.items():
                print(f"  ├ {pressure}: {pct:.1f}%")
        
        # Análise de localização
        location_dist = cluster_data['location'].value_counts(normalize=True) * 100
        print("🏠 Distribuição Casa/Fora:")
        for location, pct in location_dist.items():
            loc_name = "Casa" if location == 'h' else "Fora"
            print(f"  ├ {loc_name}: {pct:.1f}%")
    
    # Análise comparativa entre clusters
    print(f"\n📊 COMPARAÇÃO ENTRE CLUSTERS")
    print("=" * 80)
    
    cluster_summary = processed_data.groupby('cluster').agg({
        'result': lambda x: (x == 'W').mean() * 100,  # Taxa de vitória
        'goals': ['mean', 'std'],  # Gols (média e desvio)
        'teamName': 'count'  # Número de jogos
    }).round(3)
    
    cluster_summary.columns = ['Taxa_Vitoria_%', 'Gols_Media', 'Gols_Std', 'Num_Jogos']
    print(cluster_summary)
    
    # Identificar clusters com melhor defesa (menor média de gols sofridos fora)
    away_defense = processed_data[processed_data['location'] == 'a'].groupby('cluster')['goals'].mean().sort_values()
    print(f"\n🛡️ RANKING DEFENSIVO (gols sofridos fora - menor é melhor):")
    for i, (cluster, avg_goals) in enumerate(away_defense.items(), 1):
        print(f"  {i}º Cluster {cluster}: {avg_goals:.3f} gols/jogo")
    
    return processed_data

def visualize_pca(features_scaled, cluster_labels, processed_data):
    """
    Visualiza os clusters usando PCA para redução de dimensionalidade.
    
    Args:
        features_scaled (np.array): Features normalizadas
        cluster_labels (np.array): Labels dos clusters  
        processed_data (pd.DataFrame): Dados processados
    """
    print("🎨 Criando visualização PCA...")
    
    # Aplicar PCA para 2 componentes
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # Variância explicada
    explained_var = pca.explained_variance_ratio_
    print(f"📊 Variância explicada pelo PCA:")
    print(f"  ├ PC1: {explained_var[0]:.3f} ({explained_var[0]*100:.1f}%)")
    print(f"  └ PC2: {explained_var[1]:.3f} ({explained_var[1]*100:.1f}%)")
    print(f"  Total: {sum(explained_var):.3f} ({sum(explained_var)*100:.1f}%)")
    
    # Criar visualização
    plt.figure(figsize=(14, 10))
    
    # Cores para clusters
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(cluster_labels))))
    
    # Plot para cada cluster
    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        cluster_data = processed_data[cluster_mask]
        
        # Separar por localização (casa/fora)
        home_mask = cluster_data['location'] == 'h'
        away_mask = cluster_data['location'] == 'a'
        
        # Plot jogos em casa
        if home_mask.sum() > 0:
            plt.scatter(features_pca[cluster_mask][home_mask, 0], 
                       features_pca[cluster_mask][home_mask, 1],
                       c=[colors[cluster]], marker='o', s=60, alpha=0.7,
                       label=f'Cluster {cluster} - Casa' if cluster == 0 else None,
                       edgecolors='black', linewidth=0.5)
        
        # Plot jogos fora
        if away_mask.sum() > 0:
            plt.scatter(features_pca[cluster_mask][away_mask, 0], 
                       features_pca[cluster_mask][away_mask, 1],
                       c=[colors[cluster]], marker='^', s=60, alpha=0.7,
                       label=f'Cluster {cluster} - Fora' if cluster == 0 else None,
                       edgecolors='black', linewidth=0.5)
    
    plt.xlabel(f'Primeira Componente Principal (PC1) - {explained_var[0]*100:.1f}% da variância', 
               fontsize=12)
    plt.ylabel(f'Segunda Componente Principal (PC2) - {explained_var[1]*100:.1f}% da variância', 
               fontsize=12)
    plt.title('Visualização dos Clusters - Análise Defensiva de Times\n' + 
              '(Círculos = Casa, Triângulos = Fora)', fontsize=14, fontweight='bold')
    
    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Adicionar cores dos clusters
    for cluster in np.unique(cluster_labels):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors[cluster], markersize=10,
                                    label=f'Cluster {cluster}', markeredgecolor='black'))
    
    # Adicionar marcadores casa/fora
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Jogos em Casa', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=10, label='Jogos Fora', markeredgecolor='black')
    ])
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_results(processed_data):
    """
    Exporta os resultados finais para arquivo CSV.
    
    Args:
        processed_data (pd.DataFrame): Dados com clusters atribuídos
    """
    print("💾 Exportando resultados...")
    
    # Selecionar colunas para exportação
    export_columns = ['gameID', 'teamName', 'season', 'date', 'location', 
                     'cluster', 'pressure', 'result', 'awayGoals', 'homeGoals']
    
    # Verificar quais colunas existem
    available_export_cols = [col for col in export_columns if col in processed_data.columns]
    
    export_data = processed_data[available_export_cols].copy()
    
    # Salvar arquivo
    output_file = 'teamStats_clusters.csv'
    export_data.to_csv(output_file, index=False)
    
    print(f"✅ Arquivo salvo: {output_file}")
    print(f"📊 Registros exportados: {len(export_data)}")
    print(f"📋 Colunas exportadas: {list(export_data.columns)}")
    
    # Docstring para uso futuro
    docstring = """
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
    """
    
    with open('teamStats_clusters_README.md', 'w', encoding='utf-8') as f:
        f.write(docstring)
    
    return export_data

def main():
    """
    Função principal que executa todo o pipeline de clustering.
    """
    print("🚀 INICIANDO ANÁLISE DE CLUSTERING DEFENSIVO")
    print("=" * 80)
    print("Objetivo: Identificar padrões defensivos e times com melhores defesas")
    print("Foco especial: Variável 'awayGoals' e performance defensiva\n")
    
    try:
        # 1. Carregamento e merge dos dados
        merged_data = load_and_merge_data()
        
        # 2. Preparação das features
        features_scaled, scaler, feature_names, processed_data = prepare_features(merged_data)
        
        # 3. Encontrar número ótimo de clusters
        optimal_k = find_optimal_clusters(features_scaled)
        
        # 4. Treinar modelo final
        kmeans_model, cluster_labels = train_kmeans(features_scaled, optimal_k)
        
        # 5. Analisar clusters
        final_data = analyze_clusters(processed_data, cluster_labels, 
                                    features_scaled, scaler, feature_names)
        
        # 6. Visualização PCA
        visualize_pca(features_scaled, cluster_labels, processed_data)
        
        # 7. Exportar resultados
        export_data = export_results(final_data)
        
        # 8. Mostrar primeiras linhas do resultado
        print(f"\n📋 PRIMEIRAS 5 LINHAS DO DATASET COM CLUSTERS:")
        print("=" * 80)
        display_cols = ['teamName', 'date', 'location', 'result', 'cluster', 'pressure']
        available_display_cols = [col for col in display_cols if col in export_data.columns]
        print(export_data[available_display_cols].head())
        
        print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("📁 Arquivos gerados:")
        print("  ├ teamStats_clusters.csv (dados com clusters)")
        print("  ├ teamStats_clusters_README.md (documentação)")
        print("  ├ cluster_analysis.png (análise de clusters)")
        print("  └ clusters_pca_visualization.png (visualização PCA)")
        
        # Sugestões para extensões futuras
        print(f"\n💡 SUGESTÕES PARA ANÁLISES FUTURAS:")
        print("  ├ Testar Gaussian Mixture Models (GMM)")
        print("  ├ Experimentar DBSCAN para clustering baseado em densidade") 
        print("  ├ Analisar evolução temporal dos clusters")
        print("  ├ Correlacionar clusters com métricas avançadas (xG, etc.)")
        print("  └ Aplicar clustering hierárquico para análise detalhada")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()