"""
Script de Análise de Desempenho Ofensivo de Times de Futebol
=============================================================

Objetivo: Investigar quais times mais fazem gols no primeiro tempo e
quais padrões estatísticos estão relacionados a isso.

Autor: Assistente de Ciência de Dados
Data: 2025-10-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, silhouette_score, davies_bouldin_score
import warnings
import os
warnings.filterwarnings('ignore')

# Configuração de estilo para visualizações
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Diretório para salvar arquivos
OUTPUT_DIR = 'docs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_merge_data():
    """
    Carrega os datasets e realiza a junção entre teamStats e gameStats.
    
    Returns:
        pd.DataFrame: DataFrame combinado com dados de times e jogos
    """
    print("\n[1/4] Carregando dados...")
    team_stats = pd.read_csv('dataset/transformed_dataset/teamStats_transformed.csv')
    game_stats = pd.read_csv('dataset/transformed_dataset/gameStats_transformed.csv')
    print(f"✓ TeamStats: {len(team_stats)} | GameStats: {len(game_stats)}")
    
    # Garantir que gameID é numérico
    team_stats['gameID'] = pd.to_numeric(team_stats['gameID'], errors='coerce')
    game_stats['gameID'] = pd.to_numeric(game_stats['gameID'], errors='coerce')
    
    print("\n[2/4] Realizando merge...")
    # Separar times da casa (h) e visitantes (a)
    home_teams = team_stats[team_stats['location'] == 'h'].copy()
    away_teams = team_stats[team_stats['location'] == 'a'].copy()
    
    # Merge para times da casa
    home_merged = home_teams.merge(
        game_stats[['gameID', 'homeGoalsHalfTime']],
        on='gameID',
        how='inner'
    )
    home_merged['goalsHalfTime'] = home_merged['homeGoalsHalfTime']
    home_merged.drop('homeGoalsHalfTime', axis=1, inplace=True)
    
    # Merge para times visitantes
    away_merged = away_teams.merge(
        game_stats[['gameID', 'awayGoalsHalfTime']],
        on='gameID',
        how='inner'
    )
    away_merged['goalsHalfTime'] = away_merged['awayGoalsHalfTime']
    away_merged.drop('awayGoalsHalfTime', axis=1, inplace=True)
    
    # Combinar ambos
    combined = pd.concat([home_merged, away_merged], ignore_index=True)
    print(f"✓ {len(combined)} registros após junção\n")
    
    return combined


def prepare_data(df):
    """
    Prepara os dados para análise: seleciona colunas, trata valores nulos,
    converte tipos e cria variáveis derivadas.
    
    Args:
        df (pd.DataFrame): DataFrame combinado
    
    Returns:
        pd.DataFrame: DataFrame preparado para análise
    """
    print("[3/4] Preparando dados...")
    
    # Selecionar colunas de interesse
    columns_of_interest = [
        'teamName', 'season', 'date', 'location', 'pressure', 
        'ppda', 'corners', 'shots', 'shotsOnTarget', 'xGoals', 'goalsHalfTime'
    ]
    
    df_clean = df[columns_of_interest].copy()
    
    # Tratar valores nulos
    df_clean['goalsHalfTime'].fillna(0, inplace=True)
    df_clean['ppda'].fillna(df_clean['ppda'].median(), inplace=True)
    df_clean['xGoals'].fillna(0, inplace=True)
    
    # Converter colunas numéricas
    numeric_columns = ['ppda', 'corners', 'shots', 'shotsOnTarget', 'xGoals', 'goalsHalfTime']
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Converter pressure para ordinal
    pressure_mapping = {'Low Pressure': 1, 'Medium Pressure': 2, 'High Pressure': 3}
    df_clean['pressure_ordinal'] = df_clean['pressure'].map(pressure_mapping)
    df_clean['pressure_ordinal'].fillna(2, inplace=True)
    
    print(f"✓ {len(df_clean)} registros preparados\n")
    
    return df_clean


def analyze_top_scorers(df):
    """
    Analisa os times com melhor desempenho ofensivo no primeiro tempo.
    Calcula estatísticas descritivas e correlações.
    
    Args:
        df (pd.DataFrame): DataFrame preparado
    
    Returns:
        tuple: (top_teams_df, correlations_df)
    """
    print("[4/4] Analisando top scorers...\n")
    
    # Calcular estatísticas por time
    team_stats = df.groupby('teamName').agg({
        'goalsHalfTime': ['mean', 'median', 'std', 'sum'],
        'xGoals': 'mean',
        'shots': 'mean',
        'shotsOnTarget': 'mean',
        'corners': 'mean',
        'ppda': 'mean',
        'pressure_ordinal': lambda x: x.mode()[0] if len(x.mode()) > 0 else 2
    }).round(3)
    
    team_stats.columns = [
        'goalsHalfTime_mean', 'goalsHalfTime_median', 'goalsHalfTime_std', 'goalsHalfTime_total',
        'xGoals_mean', 'shots_mean', 'shotsOnTarget_mean', 'corners_mean', 'ppda_mean', 'pressure_mode'
    ]
    team_stats = team_stats.reset_index()
    team_stats = team_stats.sort_values('goalsHalfTime_mean', ascending=False)
    
    # Top 10 times
    print("=" * 80)
    print("TOP 10 TIMES - MAIOR MÉDIA DE GOLS NO 1º TEMPO")
    print("=" * 80)
    top_10 = team_stats.head(10)
    for idx, row in top_10.iterrows():
        print(f"{top_10.index.get_loc(idx)+1:2d}. {row['teamName']:25s} | "
              f"Gols: {row['goalsHalfTime_mean']:.3f} | "
              f"xG: {row['xGoals_mean']:.3f} | "
              f"Chutes: {row['shots_mean']:.1f}")
    
    # Correlações
    correlation_vars = ['goalsHalfTime', 'xGoals', 'shots', 'shotsOnTarget', 'corners', 'ppda', 'pressure_ordinal']
    correlations = df[correlation_vars].corr()['goalsHalfTime'].drop('goalsHalfTime').sort_values(ascending=False)
    
    print("\n" + "=" * 80)
    print("CORRELAÇÕES COM GOLS NO 1º TEMPO")
    print("=" * 80)
    for var, corr in correlations.items():
        print(f"{var:20s}: {corr:+.4f}")
    print()
    
    return team_stats, correlations


def plot_relationships(df, team_stats):
    """
    Cria visualizações para explorar relações entre variáveis ofensivas.
    
    Args:
        df (pd.DataFrame): DataFrame preparado
        team_stats (pd.DataFrame): Estatísticas agregadas por time
    """
    print("Gerando visualizações...")
    
    # 1. Gráfico de barras - Top 15 times
    plt.figure(figsize=(14, 8))
    top_15 = team_stats.head(15)
    plt.barh(range(len(top_15)), top_15['goalsHalfTime_mean'], color='steelblue')
    plt.yticks(range(len(top_15)), top_15['teamName'])
    plt.xlabel('Média de Gols no 1º Tempo', fontsize=12)
    plt.ylabel('Time', fontsize=12)
    plt.title('Top 15 Times - Média de Gols no Primeiro Tempo', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top15_goals_halfTime.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot - xGoals vs goalsHalfTime
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['xGoals'], df['goalsHalfTime'], 
                         c=df['pressure_ordinal'], cmap='RdYlGn_r', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('Expected Goals (xGoals)', fontsize=12)
    plt.ylabel('Gols no 1º Tempo', fontsize=12)
    plt.title('Relação entre xGoals e Gols no 1º Tempo (colorido por Pressão)', 
              fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Pressão (1=Low, 2=Medium, 3=High)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_xGoals_vs_goals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PPDA vs Gols
    plt.figure(figsize=(12, 8))
    plt.scatter(df['ppda'], df['goalsHalfTime'], alpha=0.5, color='coral', edgecolors='black', linewidth=0.5)
    plt.xlabel('PPDA (Passes Permitidos por Ação Defensiva)', fontsize=12)
    plt.ylabel('Gols no 1º Tempo', fontsize=12)
    plt.title('Relação entre PPDA e Gols no 1º Tempo', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_ppda_vs_goals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap de correlação
    correlation_vars = ['goalsHalfTime', 'xGoals', 'shots', 'shotsOnTarget', 'corners', 'ppda', 'pressure_ordinal']
    corr_matrix = df[correlation_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação - Variáveis Ofensivas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Boxplot - Top 5 vs Bottom 5
    top_5_teams = team_stats.head(5)['teamName'].tolist()
    bottom_5_teams = team_stats.tail(5)['teamName'].tolist()
    
    df_top = df[df['teamName'].isin(top_5_teams)].copy()
    df_bottom = df[df['teamName'].isin(bottom_5_teams)].copy()
    df_top['group'] = 'Top 5'
    df_bottom['group'] = 'Bottom 5'
    df_comparison = pd.concat([df_top, df_bottom])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df_comparison, x='group', y='xGoals', ax=axes[0], palette='Set2')
    axes[0].set_title('Distribuição de xGoals', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('xGoals', fontsize=11)
    
    sns.boxplot(data=df_comparison, x='group', y='shots', ax=axes[1], palette='Set2')
    axes[1].set_title('Distribuição de Chutes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Chutes', fontsize=11)
    
    plt.suptitle('Comparação: Times com Mais vs Menos Gols no 1º Tempo', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_top_vs_bottom.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 5 visualizações salvas em /{OUTPUT_DIR}\n")


def regression_analysis(df):
    """
    Realiza análise de regressão linear para identificar variáveis
    que mais influenciam os gols no primeiro tempo.
    
    Args:
        df (pd.DataFrame): DataFrame preparado
    
    Returns:
        dict: Dicionário com coeficientes e métricas do modelo
    """
    print("=" * 80)
    print("REGRESSÃO LINEAR - INFLUÊNCIA DAS VARIÁVEIS NOS GOLS")
    print("=" * 80)
    
    # Remover valores nulos
    df_reg = df[['xGoals', 'shots', 'shotsOnTarget', 'corners', 'ppda', 'goalsHalfTime']].dropna()
    
    X = df_reg[['xGoals', 'shots', 'shotsOnTarget', 'corners', 'ppda']]
    y = df_reg['goalsHalfTime']
    
    # Treinar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Predições e métricas
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Resultados
    print(f"R² Score: {r2:.4f} | Intercepto: {model.intercept_:.4f}\n")
    print("Coeficientes (ordem de importância):")
    
    coefficients = dict(zip(X.columns, model.coef_))
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for var, coef in sorted_coefs:
        print(f"  {var:20s}: {coef:+.6f}")
    print()
    
    return {
        'model': model,
        'r2': r2,
        'coefficients': coefficients,
        'feature_names': X.columns.tolist()
    }


def calculate_dunn_index(X, labels):
    """
    Calcula o Índice de Dunn para avaliar a qualidade dos clusters.
    Dunn = (menor distância entre clusters) / (maior diâmetro de cluster)
    Valores maiores indicam melhor separação.
    
    Args:
        X (np.array): Dados normalizados
        labels (np.array): Labels dos clusters
    
    Returns:
        float: Índice de Dunn
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Calcular centroides
    centroids = np.array([X[labels == i].mean(axis=0) for i in unique_labels])
    
    # Calcular menor distância entre centroides (inter-cluster)
    min_inter_cluster_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_inter_cluster_dist:
                min_inter_cluster_dist = dist
    
    # Calcular maior diâmetro de cluster (intra-cluster)
    max_intra_cluster_dist = 0
    for i in unique_labels:
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            # Calcular todas as distâncias dentro do cluster
            for p1 in cluster_points:
                for p2 in cluster_points:
                    dist = np.linalg.norm(p1 - p2)
                    if dist > max_intra_cluster_dist:
                        max_intra_cluster_dist = dist
    
    if max_intra_cluster_dist == 0:
        return 0
    
    return min_inter_cluster_dist / max_intra_cluster_dist


def evaluate_clustering_quality(X_scaled, k_range=range(2, 11)):
    """
    Avalia a qualidade dos clusters usando múltiplas métricas e diferentes valores de K.
    Gera o gráfico do método Elbow e calcula métricas de qualidade.
    
    Args:
        X_scaled (np.array): Dados normalizados
        k_range (range): Range de valores de K para testar
    
    Returns:
        tuple: (optimal_k, metrics_df)
    """
    print("\n" + "=" * 80)
    print("AVALIAÇÃO DA QUALIDADE DOS CLUSTERS")
    print("=" * 80)
    
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    dunn_indices = []
    
    print("\nCalculando métricas para diferentes valores de K...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Inércia (para método Elbow)
        inertias.append(kmeans.inertia_)
        
        # Silhouette Score (maior = melhor, range: -1 a 1)
        silhouette = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette)
        
        # Davies-Bouldin Index (menor = melhor)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        davies_bouldin_scores.append(davies_bouldin)
        
        # Dunn Index (maior = melhor)
        dunn = calculate_dunn_index(X_scaled, labels)
        dunn_indices.append(dunn)
        
        print(f"K={k}: Silhouette={silhouette:.4f} | Davies-Bouldin={davies_bouldin:.4f} | Dunn={dunn:.4f}")
    
    # Criar DataFrame com métricas
    metrics_df = pd.DataFrame({
        'K': list(k_range),
        'Inertia': inertias,
        'Silhouette_Score': silhouette_scores,
        'Davies_Bouldin_Index': davies_bouldin_scores,
        'Dunn_Index': dunn_indices
    })
    
    # Salvar métricas em CSV
    metrics_file = os.path.join(OUTPUT_DIR, 'cluster_quality_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    print(f"\n✓ Métricas salvas: {metrics_file}")
    
    # 1. GRÁFICO DE ELBOW
    plt.figure(figsize=(12, 8))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Inércia (Soma das Distâncias Quadráticas)', fontsize=12)
    plt.title('Método Elbow - Determinação do K Ótimo', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. GRÁFICO COM TODAS AS MÉTRICAS
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Silhouette Score
    axes[0, 0].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[0, 0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0, 0].set_title('Silhouette Score (Maior = Melhor)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(k_range)
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    axes[0, 0].axvline(best_silhouette_k, color='red', linestyle='--', alpha=0.7, label=f'Melhor K={best_silhouette_k}')
    axes[0, 0].legend()
    
    # Davies-Bouldin Index
    axes[0, 1].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[0, 1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[0, 1].set_title('Davies-Bouldin Index (Menor = Melhor)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(k_range)
    best_db_k = k_range[np.argmin(davies_bouldin_scores)]
    axes[0, 1].axvline(best_db_k, color='red', linestyle='--', alpha=0.7, label=f'Melhor K={best_db_k}')
    axes[0, 1].legend()
    
    # Dunn Index
    axes[1, 0].plot(k_range, dunn_indices, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[1, 0].set_ylabel('Dunn Index', fontsize=11)
    axes[1, 0].set_title('Dunn Index (Maior = Melhor)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xticks(k_range)
    best_dunn_k = k_range[np.argmax(dunn_indices)]
    axes[1, 0].axvline(best_dunn_k, color='red', linestyle='--', alpha=0.7, label=f'Melhor K={best_dunn_k}')
    axes[1, 0].legend()
    
    # Comparação normalizada
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_metrics = pd.DataFrame({
        'K': k_range,
        'Silhouette': scaler.fit_transform(np.array(silhouette_scores).reshape(-1, 1)).flatten(),
        'Davies-Bouldin (inv)': 1 - scaler.fit_transform(np.array(davies_bouldin_scores).reshape(-1, 1)).flatten(),
        'Dunn': scaler.fit_transform(np.array(dunn_indices).reshape(-1, 1)).flatten()
    })
    
    axes[1, 1].plot(k_range, normalized_metrics['Silhouette'], 'go-', linewidth=2, markersize=6, label='Silhouette')
    axes[1, 1].plot(k_range, normalized_metrics['Davies-Bouldin (inv)'], 'ro-', linewidth=2, markersize=6, label='Davies-Bouldin (inv)')
    axes[1, 1].plot(k_range, normalized_metrics['Dunn'], 'mo-', linewidth=2, markersize=6, label='Dunn')
    axes[1, 1].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[1, 1].set_ylabel('Métricas Normalizadas (0-1)', fontsize=11)
    axes[1, 1].set_title('Comparação de Todas as Métricas', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xticks(k_range)
    axes[1, 1].legend()
    
    plt.suptitle('Análise de Qualidade dos Clusters', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_quality_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determinar K ótimo baseado em múltiplas métricas
    optimal_k = best_silhouette_k
    
    print("\n" + "-" * 80)
    print("RECOMENDAÇÕES:")
    print(f"  Melhor K por Silhouette Score:      {best_silhouette_k}")
    print(f"  Melhor K por Davies-Bouldin Index:  {best_db_k}")
    print(f"  Melhor K por Dunn Index:            {best_dunn_k}")
    print(f"  K Recomendado (Silhouette):         {optimal_k}")
    print("-" * 80 + "\n")
    
    print(f"✓ Gráficos salvos em /{OUTPUT_DIR}\n")
    
    return optimal_k, metrics_df


def optional_cluster_analysis(df, team_stats):
    """
    Realiza análise de clusterização (KMeans) para identificar
    estilos ofensivos distintos entre os times.
    Inclui avaliação completa de qualidade dos clusters.
    
    Args:
        df (pd.DataFrame): DataFrame preparado
        team_stats (pd.DataFrame): Estatísticas agregadas por time
    
    Returns:
        pd.DataFrame: team_stats com coluna de cluster adicionada
    """
    print("=" * 80)
    print("CLUSTERIZAÇÃO - ESTILOS OFENSIVOS")
    print("=" * 80)
    
    # Selecionar features para clustering
    features = ['xGoals_mean', 'shots_mean', 'shotsOnTarget_mean', 'corners_mean', 'ppda_mean']
    X_cluster = team_stats[features].copy()
    
    # Padronizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Avaliar qualidade dos clusters e determinar K ótimo
    optimal_k, metrics_df = evaluate_clustering_quality(X_scaled, k_range=range(2, 11))
    
    # Usar K ótimo para clustering final
    print(f"Executando KMeans com K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    team_stats['cluster_label'] = kmeans.fit_predict(X_scaled)
    
    # Calcular métricas finais
    final_silhouette = silhouette_score(X_scaled, team_stats['cluster_label'])
    final_davies_bouldin = davies_bouldin_score(X_scaled, team_stats['cluster_label'])
    final_dunn = calculate_dunn_index(X_scaled, team_stats['cluster_label'].values)
    
    print("\n" + "=" * 80)
    print(f"MÉTRICAS DO MODELO FINAL (K={optimal_k})")
    print("=" * 80)
    print(f"Silhouette Score:      {final_silhouette:.4f} (range: -1 a 1, maior = melhor)")
    print(f"Davies-Bouldin Index:  {final_davies_bouldin:.4f} (menor = melhor)")
    print(f"Dunn Index:            {final_dunn:.4f} (maior = melhor)")
    print("=" * 80)
    
    # Analisar características dos clusters
    print("\nCARACTERÍSTICAS DOS CLUSTERS:\n")
    for cluster_id in range(optimal_k):
        cluster_teams = team_stats[team_stats['cluster_label'] == cluster_id]
        print(f"CLUSTER {cluster_id} ({len(cluster_teams)} times):")
        print(f"  Gols 1º tempo: {cluster_teams['goalsHalfTime_mean'].mean():.3f}")
        print(f"  xGoals:        {cluster_teams['xGoals_mean'].mean():.3f}")
        print(f"  Chutes:        {cluster_teams['shots_mean'].mean():.1f}")
        print(f"  PPDA:          {cluster_teams['ppda_mean'].mean():.2f}")
        print(f"  Exemplos: {', '.join(cluster_teams['teamName'].head(3).tolist())}\n")
    
    # Visualizar clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(team_stats['xGoals_mean'], team_stats['goalsHalfTime_mean'],
                         c=team_stats['cluster_label'], cmap='viridis', 
                         s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    plt.xlabel('xGoals Médio', fontsize=12)
    plt.ylabel('Gols no 1º Tempo (Média)', fontsize=12)
    plt.title(f'Clusters de Estilos Ofensivos (K={optimal_k})', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_offensive_styles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizações salvas em /{OUTPUT_DIR}\n")
    
    return team_stats


def export_results(team_stats):
    """
    Exporta os resultados da análise para arquivo CSV.
    
    Args:
        team_stats (pd.DataFrame): Estatísticas agregadas por time com clusters
    """
    # Selecionar colunas para exportação
    export_columns = [
        'teamName', 'goalsHalfTime_mean', 'xGoals_mean', 'shots_mean',
        'shotsOnTarget_mean', 'ppda_mean', 'pressure_mode', 'cluster_label'
    ]
    
    export_df = team_stats[export_columns].copy()
    export_df = export_df.sort_values('goalsHalfTime_mean', ascending=False)
    
    # Salvar
    output_file = os.path.join(OUTPUT_DIR, 'offensive_analysis.csv')
    export_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Exportado: {output_file} ({len(export_df)} times)\n")


def main():
    """
    Função principal que orquestra todo o pipeline de análise.
    """
    print("\n" + "=" * 80)
    print("ANÁLISE DE DESEMPENHO OFENSIVO - FUTEBOL")
    print("=" * 80 + "\n")
    
    # 1. Carregar e juntar dados
    combined_df = load_and_merge_data()
    
    # 2. Preparar dados
    prepared_df = prepare_data(combined_df)
    
    # 3. Análise dos top scorers
    team_stats, correlations = analyze_top_scorers(prepared_df)
    
    # 4. Visualizações
    plot_relationships(prepared_df, team_stats)
    
    # 5. Regressão linear
    regression_results = regression_analysis(prepared_df)
    
    # 6. Clusterização
    team_stats = optional_cluster_analysis(prepared_df, team_stats)
    
    # 7. Exportar resultados
    export_results(team_stats)
    
    # Resumo final
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Registros analisados: {len(prepared_df)}")
    print(f"Times únicos: {prepared_df['teamName'].nunique()}")
    print(f"Temporadas: {sorted(prepared_df['season'].unique())}")
    print(f"R² regressão: {regression_results['r2']:.4f}")
    print(f"\nArquivos gerados em /{OUTPUT_DIR}:")
    print("  CSV:")
    print("    - offensive_analysis.csv")
    print("    - cluster_quality_metrics.csv")
    print("  Gráficos:")
    print("    - top15_goals_halfTime.png")
    print("    - scatter_xGoals_vs_goals.png")
    print("    - scatter_ppda_vs_goals.png")
    print("    - correlation_heatmap.png")
    print("    - boxplot_top_vs_bottom.png")
    print("    - elbow_method.png")
    print("    - cluster_quality_metrics.png")
    print("    - clusters_offensive_styles.png")
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
