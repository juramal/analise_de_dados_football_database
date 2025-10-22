"""
Script para Gerar Gráficos das Métricas de Avaliação de Clustering

Este script gera especificamente os gráficos solicitados:
- Silhouette Score
- Índice Davies-Bouldin  
- Índice Dunn

Autor: Assistente de Ciência de Dados
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, cdist
import warnings
warnings.filterwarnings('ignore')

def calculate_dunn_index(X, labels):
    """
    Calcula o Índice de Dunn para avaliação de clustering.
    Dunn Index = min(distância entre clusters) / max(diâmetro intra-cluster)
    Valores maiores indicam melhor clustering.
    """
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

def load_and_prepare_data():
    """Carrega e prepara os dados para análise"""
    print("🔄 Carregando e preparando dados...")
    
    # Carregar datasets
    team_stats = pd.read_csv('dataset/transformed_dataset/teamStats_transformed.csv')
    game_stats = pd.read_csv('dataset/transformed_dataset/gameStats_transformed.csv')
    
    # Garantir que gameID seja numérico
    team_stats['gameID'] = pd.to_numeric(team_stats['gameID'], errors='coerce')
    game_stats['gameID'] = pd.to_numeric(game_stats['gameID'], errors='coerce')
    
    # Remover registros com gameID inválido
    team_stats = team_stats.dropna(subset=['gameID'])
    game_stats = game_stats.dropna(subset=['gameID'])
    
    # Fazer merge das tabelas
    merged_data = team_stats.merge(game_stats, on='gameID', how='inner', suffixes=('', '_game'))
    
    # Remover duplicatas
    merged_data = merged_data.drop_duplicates()
    
    # Remover colunas redundantes
    columns_to_drop = ['season_game', 'date_game']
    existing_cols_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
    merged_data = merged_data.drop(columns=existing_cols_to_drop)
    
    # Criar variável binária para localização
    merged_data['location_bin'] = (merged_data['location'] == 'h').astype(int)
    
    # Selecionar features para clustering
    defensive_features = [
        'goals', 'xGoals', 'shots', 'shotsOnTarget', 'deep', 'ppda', 
        'fouls', 'corners', 'yellowCards', 'redCards',
        'homeProbability', 'drawProbability', 'awayProbability',
        'B365H', 'B365D', 'B365A', 'location_bin'
    ]
    
    # Verificar quais features existem no dataset
    available_features = [col for col in defensive_features if col in merged_data.columns]
    
    # Criar dataset apenas com features numéricas
    features_data = merged_data[available_features].copy()
    
    # Tratar valores infinitos e ausentes
    features_data = features_data.replace([np.inf, -np.inf], np.nan)
    
    # Preencher valores ausentes com a mediana
    for col in features_data.columns:
        if features_data[col].isnull().sum() > 0:
            median_val = features_data[col].median()
            features_data[col].fillna(median_val, inplace=True)
    
    # Aplicar StandardScaler para normalização
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_data)
    
    print(f"✅ Dados preparados: {features_scaled.shape[0]} amostras, {features_scaled.shape[1]} features")
    
    return features_scaled, available_features

def generate_clustering_metrics_plots(features_scaled, max_clusters=10):
    """
    Gera gráficos das métricas de avaliação de clustering solicitadas
    """
    print("📊 Gerando gráficos das métricas de clustering...")
    
    # Range de clusters para testar
    k_range = range(2, max_clusters + 1)
    
    # Listas para armazenar métricas
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    dunn_indices = []
    
    print("🔄 Calculando métricas para diferentes números de clusters...")
    
    for k in k_range:
        print(f"  └ Calculando para k={k}...")
        
        # Treinar KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        inertias.append(kmeans.inertia_)
        
        # Silhouette Score (maior é melhor)
        sil_score = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        
        # Davies-Bouldin Score (menor é melhor)
        db_score = davies_bouldin_score(features_scaled, cluster_labels)
        davies_bouldin_scores.append(db_score)
        
        # Dunn Index (maior é melhor)
        dunn_score = calculate_dunn_index(features_scaled, cluster_labels)
        dunn_indices.append(dunn_score)
        
        print(f"    ├ Silhouette: {sil_score:.3f}")
        print(f"    ├ Davies-Bouldin: {db_score:.3f}")
        print(f"    └ Dunn Index: {dunn_score:.3f}")
    
    # Encontrar valores ótimos
    optimal_silhouette_k = k_range[np.argmax(silhouette_scores)]
    optimal_db_k = k_range[np.argmin(davies_bouldin_scores)]
    optimal_dunn_k = k_range[np.argmax(dunn_indices)]
    
    print(f"\n📈 Resultados das Métricas:")
    print(f"  ├ Melhor Silhouette Score: k={optimal_silhouette_k} ({max(silhouette_scores):.3f})")
    print(f"  ├ Melhor Davies-Bouldin: k={optimal_db_k} ({min(davies_bouldin_scores):.3f})")
    print(f"  └ Melhor Dunn Index: k={optimal_dunn_k} ({max(dunn_indices):.3f})")
    
    # 1. GRÁFICO INDIVIDUAL PARA CADA MÉTRICA
    create_individual_metrics_plots(k_range, silhouette_scores, davies_bouldin_scores, 
                                   dunn_indices, optimal_silhouette_k, optimal_db_k, optimal_dunn_k)
    
    # 2. GRÁFICO COMPARATIVO DAS TRÊS MÉTRICAS
    create_combined_metrics_plot(k_range, silhouette_scores, davies_bouldin_scores, 
                                dunn_indices, optimal_silhouette_k, optimal_db_k, optimal_dunn_k)
    
    # 3. GRÁFICO COM MÉTRICAS NORMALIZADAS
    create_normalized_comparison(k_range, silhouette_scores, davies_bouldin_scores, dunn_indices)
    
    # 4. TABELA RESUMO DAS MÉTRICAS
    create_metrics_summary_table(k_range, silhouette_scores, davies_bouldin_scores, dunn_indices)
    
    return {
        'k_range': k_range,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'dunn_indices': dunn_indices,
        'optimal_k': optimal_silhouette_k
    }

def create_individual_metrics_plots(k_range, sil_scores, db_scores, dunn_scores, 
                                   opt_sil_k, opt_db_k, opt_dunn_k):
    """Cria gráficos individuais para cada métrica"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Métricas de Avaliação de Clustering para Análise Defensiva de Futebol', 
                 fontsize=16, fontweight='bold')
    
    # 1. Silhouette Score
    ax1 = axes[0]
    ax1.plot(k_range, sil_scores, 'ro-', linewidth=3, markersize=10, alpha=0.8)
    ax1.axhline(y=max(sil_scores), color='green', linestyle='--', alpha=0.7, 
                label=f'Máximo: {max(sil_scores):.3f}')
    ax1.axvline(x=opt_sil_k, color='green', linestyle='--', alpha=0.7, 
                label=f'Ótimo: k={opt_sil_k}')
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Silhouette Score\n(Maior = Melhor)', fontsize=14, fontweight='bold', color='darkred')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(k_range)
    
    # Adicionar valores nos pontos
    for i, (k, score) in enumerate(zip(k_range, sil_scores)):
        ax1.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, fontweight='bold')
    
    # 2. Davies-Bouldin Score
    ax2 = axes[1]
    ax2.plot(k_range, db_scores, 'go-', linewidth=3, markersize=10, alpha=0.8)
    ax2.axhline(y=min(db_scores), color='red', linestyle='--', alpha=0.7, 
                label=f'Mínimo: {min(db_scores):.3f}')
    ax2.axvline(x=opt_db_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Ótimo: k={opt_db_k}')
    ax2.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Davies-Bouldin Score', fontsize=12, fontweight='bold')
    ax2.set_title('Índice Davies-Bouldin\n(Menor = Melhor)', fontsize=14, fontweight='bold', color='darkgreen')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(k_range)
    
    # Adicionar valores nos pontos
    for i, (k, score) in enumerate(zip(k_range, db_scores)):
        ax2.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, fontweight='bold')
    
    # 3. Dunn Index
    ax3 = axes[2]
    ax3.plot(k_range, dunn_scores, 'mo-', linewidth=3, markersize=10, alpha=0.8)
    ax3.axhline(y=max(dunn_scores), color='orange', linestyle='--', alpha=0.7, 
                label=f'Máximo: {max(dunn_scores):.3f}')
    ax3.axvline(x=opt_dunn_k, color='orange', linestyle='--', alpha=0.7, 
                label=f'Ótimo: k={opt_dunn_k}')
    ax3.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Dunn Index', fontsize=12, fontweight='bold')
    ax3.set_title('Índice Dunn\n(Maior = Melhor)', fontsize=14, fontweight='bold', color='purple')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xticks(k_range)
    
    # Adicionar valores nos pontos
    for i, (k, score) in enumerate(zip(k_range, dunn_scores)):
        ax3.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metricas_clustering_individuais.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráficos individuais salvos: metricas_clustering_individuais.png")

def create_combined_metrics_plot(k_range, sil_scores, db_scores, dunn_scores, 
                                opt_sil_k, opt_db_k, opt_dunn_k):
    """Cria gráfico combinado das três métricas"""
    
    plt.figure(figsize=(14, 8))
    
    # Plot das três métricas (normalizar Davies-Bouldin invertendo para comparação)
    plt.plot(k_range, sil_scores, 'ro-', linewidth=3, markersize=8, 
             label='Silhouette Score', alpha=0.8)
    
    # Inverter Davies-Bouldin para visualização (menor é melhor, então 1-valor normalizado)
    db_normalized = 1 - np.array(db_scores) / max(db_scores)
    plt.plot(k_range, db_normalized, 'go-', linewidth=3, markersize=8, 
             label='Davies-Bouldin (invertido)', alpha=0.8)
    
    plt.plot(k_range, dunn_scores, 'mo-', linewidth=3, markersize=8, 
             label='Dunn Index', alpha=0.8)
    
    # Marcar pontos ótimos
    plt.axvline(x=opt_sil_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Ótimo Silhouette (k={opt_sil_k})')
    plt.axvline(x=opt_db_k, color='green', linestyle='--', alpha=0.7, 
                label=f'Ótimo Davies-Bouldin (k={opt_db_k})')
    plt.axvline(x=opt_dunn_k, color='magenta', linestyle='--', alpha=0.7, 
                label=f'Ótimo Dunn (k={opt_dunn_k})')
    
    plt.xlabel('Número de Clusters (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor da Métrica', fontsize=14, fontweight='bold')
    plt.title('Comparação das Métricas de Avaliação de Clustering\n' +
              'Análise Defensiva no Futebol', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xticks(k_range)
    
    # Adicionar texto explicativo
    plt.text(0.02, 0.98, 
             'Interpretação:\n• Silhouette: Maior = Melhor\n• Davies-Bouldin: Menor = Melhor (invertido no gráfico)\n• Dunn Index: Maior = Melhor', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('metricas_clustering_comparativo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráfico comparativo salvo: metricas_clustering_comparativo.png")

def create_normalized_comparison(k_range, sil_scores, db_scores, dunn_scores):
    """Cria gráfico com métricas normalizadas para comparação direta"""
    
    # Normalizar todas as métricas para escala 0-1
    def normalize(values, invert=False):
        values = np.array(values)
        normalized = (values - values.min()) / (values.max() - values.min())
        return 1 - normalized if invert else normalized
    
    sil_norm = normalize(sil_scores)
    db_norm = normalize(db_scores, invert=True)  # Inverter pois menor é melhor
    dunn_norm = normalize(dunn_scores)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(k_range, sil_norm, 'ro-', linewidth=3, markersize=8, 
             label='Silhouette Score (norm.)', alpha=0.8)
    plt.plot(k_range, db_norm, 'go-', linewidth=3, markersize=8, 
             label='Davies-Bouldin (norm. inv.)', alpha=0.8)
    plt.plot(k_range, dunn_norm, 'mo-', linewidth=3, markersize=8, 
             label='Dunn Index (norm.)', alpha=0.8)
    
    # Calcular média das métricas normalizadas
    avg_metrics = (sil_norm + db_norm + dunn_norm) / 3
    plt.plot(k_range, avg_metrics, 'ko--', linewidth=2, markersize=6, 
             label='Média das Métricas', alpha=0.7)
    
    # Marcar melhor k pela média
    best_k_avg = k_range[np.argmax(avg_metrics)]
    plt.axvline(x=best_k_avg, color='black', linestyle=':', alpha=0.8, 
                label=f'Melhor k (média): {best_k_avg}')
    
    plt.xlabel('Número de Clusters (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Normalizado (0-1)', fontsize=14, fontweight='bold')
    plt.title('Métricas de Clustering Normalizadas\n' + 
              'Todas as métricas ajustadas para "Maior = Melhor"', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(k_range)
    plt.ylim(0, 1.1)
    
    # Adicionar valores da média nos pontos
    for i, (k, avg) in enumerate(zip(k_range, avg_metrics)):
        plt.annotate(f'{avg:.2f}', (k, avg), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metricas_clustering_normalizadas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráfico normalizado salvo: metricas_clustering_normalizadas.png")
    print(f"📊 Melhor k pela média das métricas: {best_k_avg}")

def create_metrics_summary_table(k_range, sil_scores, db_scores, dunn_scores):
    """Cria tabela resumo das métricas"""
    
    print(f"\n📊 TABELA RESUMO DAS MÉTRICAS DE CLUSTERING")
    print("=" * 80)
    print(f"{'k':>3} | {'Silhouette':>10} | {'Davies-Bouldin':>13} | {'Dunn Index':>10} | {'Ranking':>8}")
    print("-" * 80)
    
    # Calcular rankings (1 = melhor)
    sil_ranks = np.argsort(np.argsort(sil_scores)[::-1]) + 1  # Maior é melhor
    db_ranks = np.argsort(np.argsort(db_scores)) + 1  # Menor é melhor
    dunn_ranks = np.argsort(np.argsort(dunn_scores)[::-1]) + 1  # Maior é melhor
    
    # Ranking combinado (soma dos rankings individuais)
    combined_ranks = sil_ranks + db_ranks + dunn_ranks
    overall_ranks = np.argsort(np.argsort(combined_ranks)) + 1
    
    for i, k in enumerate(k_range):
        print(f"{k:>3} | {sil_scores[i]:>10.3f} | {db_scores[i]:>13.3f} | {dunn_scores[i]:>10.3f} | {overall_ranks[i]:>8}")
    
    # Identificar o melhor k
    best_k_idx = np.argmin(combined_ranks)
    best_k = k_range[best_k_idx]
    
    print("-" * 80)
    print(f"🏆 MELHOR k PELO RANKING COMBINADO: {best_k}")
    print(f"   ├ Silhouette Score: {sil_scores[best_k_idx]:.3f}")
    print(f"   ├ Davies-Bouldin Score: {db_scores[best_k_idx]:.3f}")
    print(f"   └ Dunn Index: {dunn_scores[best_k_idx]:.3f}")

def main():
    """Função principal"""
    print("🎯 GERAÇÃO DE GRÁFICOS DAS MÉTRICAS DE CLUSTERING")
    print("=" * 70)
    print("📊 Métricas a serem analisadas:")
    print("  ├ Silhouette Score (coesão interna vs separação)")
    print("  ├ Índice Davies-Bouldin (compactação vs separação)")
    print("  └ Índice Dunn (separação mínima vs diâmetro máximo)")
    print()
    
    try:
        # 1. Carregar e preparar dados
        features_scaled, feature_names = load_and_prepare_data()
        
        # 2. Gerar gráficos das métricas
        results = generate_clustering_metrics_plots(features_scaled, max_clusters=10)
        
        print(f"\n🎯 RESUMO EXECUTIVO")
        print("=" * 50)
        print(f"📊 Melhor k por métrica:")
        print(f"  ├ Silhouette Score: k={results['k_range'][np.argmax(results['silhouette_scores'])]}")
        print(f"  ├ Davies-Bouldin: k={results['k_range'][np.argmin(results['davies_bouldin_scores'])]}")
        print(f"  └ Dunn Index: k={results['k_range'][np.argmax(results['dunn_indices'])]}")
        
        print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("📁 Arquivos gerados:")
        print("  ├ metricas_clustering_individuais.png")
        print("  ├ metricas_clustering_comparativo.png")
        print("  └ metricas_clustering_normalizadas.png")
        
        print(f"\n💡 INTERPRETAÇÃO DAS MÉTRICAS:")
        print("  ├ Silhouette Score: Mede quão bem separados estão os clusters")
        print("  ├ Davies-Bouldin: Razão entre dispersão intra e inter-cluster")
        print("  └ Dunn Index: Razão entre separação mínima e diâmetro máximo")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()