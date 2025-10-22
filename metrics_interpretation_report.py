"""
Relatório de Análise das Métricas de Clustering

Este script gera um relatório detalhado interpretando os resultados 
das métricas Silhouette Score, Davies-Bouldin e Dunn Index.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_metrics_interpretation_report():
    """
    Cria um relatório interpretativo das métricas de clustering
    """
    
    print("📋 RELATÓRIO DETALHADO DAS MÉTRICAS DE CLUSTERING")
    print("=" * 80)
    print("Análise Defensiva de Times de Futebol - Dataset GameStats")
    print("=" * 80)
    
    # Dados das métricas obtidos do resultado anterior
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    silhouette_scores = [0.168, 0.140, 0.142, 0.154, 0.138, 0.133, 0.124, 0.119, 0.118]
    davies_bouldin_scores = [2.154, 2.058, 2.025, 1.880, 1.833, 1.832, 1.804, 1.901, 1.826]
    dunn_indices = [0.024, 0.019, 0.013, 0.013, 0.019, 0.020, 0.023, 0.025, 0.021]
    
    print("\n🔍 1. ANÁLISE DO SILHOUETTE SCORE")
    print("-" * 50)
    print("O Silhouette Score varia de -1 a +1:")
    print("• Valores próximos a +1: Clusters bem definidos e separados")
    print("• Valores próximos a 0: Clusters sobrepostos ou no limite")
    print("• Valores negativos: Pontos podem estar no cluster errado")
    print()
    
    best_sil_k = k_values[np.argmax(silhouette_scores)]
    best_sil_score = max(silhouette_scores)
    
    print(f"📊 Resultados Silhouette Score:")
    print(f"• Melhor resultado: k={best_sil_k} com score {best_sil_score:.3f}")
    print(f"• Interpretação: Score moderado, indica clusters razoavelmente definidos")
    print(f"• Tendência: Scores diminuem com o aumento de k (comum em datasets reais)")
    
    if best_sil_score > 0.7:
        interpretation = "Excelente separação"
    elif best_sil_score > 0.5:
        interpretation = "Boa separação"
    elif best_sil_score > 0.25:
        interpretation = "Separação moderada"
    else:
        interpretation = "Separação fraca"
    
    print(f"• Qualidade da separação: {interpretation}")
    
    print("\n🔍 2. ANÁLISE DO ÍNDICE DAVIES-BOULDIN")
    print("-" * 50)
    print("O Índice Davies-Bouldin mede a razão média entre dispersão intra-cluster")
    print("e separação inter-cluster. Valores menores indicam melhor clustering:")
    print("• Valores baixos (< 1): Clusters bem compactos e separados")
    print("• Valores altos (> 2): Clusters podem estar muito dispersos")
    print()
    
    best_db_k = k_values[np.argmin(davies_bouldin_scores)]
    best_db_score = min(davies_bouldin_scores)
    
    print(f"📊 Resultados Davies-Bouldin:")
    print(f"• Melhor resultado: k={best_db_k} com score {best_db_score:.3f}")
    print(f"• Interpretação: Score indica clusters moderadamente compactos")
    print(f"• Tendência: Melhora com k=5-8, depois se deteriora")
    
    if best_db_score < 1.0:
        db_interpretation = "Clusters muito bem definidos"
    elif best_db_score < 1.5:
        db_interpretation = "Clusters bem definidos"
    elif best_db_score < 2.0:
        db_interpretation = "Clusters moderadamente definidos"
    else:
        db_interpretation = "Clusters pouco definidos"
    
    print(f"• Qualidade dos clusters: {db_interpretation}")
    
    print("\n🔍 3. ANÁLISE DO ÍNDICE DUNN")
    print("-" * 50)
    print("O Índice Dunn mede a razão entre a mínima separação inter-cluster")
    print("e o máximo diâmetro intra-cluster. Valores maiores são melhores:")
    print("• Valores altos: Clusters bem separados e compactos")
    print("• Valores baixos: Clusters podem estar próximos ou dispersos")
    print()
    
    best_dunn_k = k_values[np.argmax(dunn_indices)]
    best_dunn_score = max(dunn_indices)
    
    print(f"📊 Resultados Índice Dunn:")
    print(f"• Melhor resultado: k={best_dunn_k} com score {best_dunn_score:.3f}")
    print(f"• Interpretação: Scores baixos indicam clusters próximos")
    print(f"• Tendência: Variação pequena entre diferentes k")
    
    if best_dunn_score > 0.1:
        dunn_interpretation = "Excelente separação"
    elif best_dunn_score > 0.05:
        dunn_interpretation = "Boa separação" 
    elif best_dunn_score > 0.02:
        dunn_interpretation = "Separação moderada"
    else:
        dunn_interpretation = "Separação fraca"
    
    print(f"• Qualidade da separação: {dunn_interpretation}")
    
    print("\n📊 4. CONSENSO E RECOMENDAÇÕES")
    print("-" * 50)
    
    # Análise de consenso
    print("📋 Resumo por métrica:")
    print(f"• Silhouette Score recomenda: k={best_sil_k}")
    print(f"• Davies-Bouldin recomenda: k={best_db_k}")
    print(f"• Dunn Index recomenda: k={best_dunn_k}")
    
    # Calcular consenso ponderado
    # Normalizar e inverter Davies-Bouldin
    sil_norm = np.array(silhouette_scores) / max(silhouette_scores)
    db_norm = min(davies_bouldin_scores) / np.array(davies_bouldin_scores)  # Inverter
    dunn_norm = np.array(dunn_indices) / max(dunn_indices)
    
    # Média ponderada (pode ajustar os pesos conforme necessário)
    weights = [0.4, 0.35, 0.25]  # Silhouette, Davies-Bouldin, Dunn
    consensus_scores = (weights[0] * sil_norm + 
                       weights[1] * db_norm + 
                       weights[2] * dunn_norm)
    
    best_consensus_k = k_values[np.argmax(consensus_scores)]
    
    print(f"\n🎯 RECOMENDAÇÃO FINAL:")
    print(f"• k ótimo pelo consenso ponderado: k={best_consensus_k}")
    print(f"• Este valor balanceia todas as três métricas")
    
    # Justificativa da escolha
    print(f"\n💡 JUSTIFICATIVA TÉCNICA:")
    if best_consensus_k == 2:
        print("• k=2 oferece a melhor separação geral (Silhouette)")
        print("• Adequado para análise defensiva: 'Defesas Sólidas' vs 'Defesas Fracas'")
        print("• Interpretação mais simples e prática para tomada de decisão")
    elif best_consensus_k <= 4:
        print(f"• k={best_consensus_k} oferece boa granularidade sem perder interpretabilidade")
        print("• Permite identificar padrões defensivos mais específicos")
    else:
        print(f"• k={best_consensus_k} pode oferecer insights mais detalhados")
        print("• Requer análise cuidadosa para evitar overfitting")
    
    print(f"\n🏈 APLICAÇÃO NO CONTEXTO FUTEBOLÍSTICO:")
    print("• Times podem ser agrupados por padrões defensivos similares")
    print("• Útil para: scouting, análise tática, comparação de performance")
    print("• Permite identificar times com características defensivas únicas")
    
    return {
        'best_silhouette_k': best_sil_k,
        'best_davies_bouldin_k': best_db_k, 
        'best_dunn_k': best_dunn_k,
        'consensus_k': best_consensus_k,
        'consensus_scores': consensus_scores
    }

def create_comparison_table():
    """
    Cria tabela comparativa detalhada das métricas
    """
    
    print(f"\n📊 5. TABELA COMPARATIVA DETALHADA")
    print("=" * 100)
    
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    silhouette_scores = [0.168, 0.140, 0.142, 0.154, 0.138, 0.133, 0.124, 0.119, 0.118]
    davies_bouldin_scores = [2.154, 2.058, 2.025, 1.880, 1.833, 1.832, 1.804, 1.901, 1.826]
    dunn_indices = [0.024, 0.019, 0.013, 0.013, 0.019, 0.020, 0.023, 0.025, 0.021]
    
    # Criar DataFrame para melhor visualização
    df = pd.DataFrame({
        'k': k_values,
        'Silhouette': silhouette_scores,
        'Davies_Bouldin': davies_bouldin_scores,
        'Dunn_Index': dunn_indices
    })
    
    # Calcular rankings
    df['Rank_Silhouette'] = df['Silhouette'].rank(ascending=False).astype(int)
    df['Rank_Davies_Bouldin'] = df['Davies_Bouldin'].rank(ascending=True).astype(int)  # Menor é melhor
    df['Rank_Dunn'] = df['Dunn_Index'].rank(ascending=False).astype(int)
    df['Rank_Total'] = df['Rank_Silhouette'] + df['Rank_Davies_Bouldin'] + df['Rank_Dunn']
    df['Rank_Final'] = df['Rank_Total'].rank(ascending=True).astype(int)
    
    print(f"{'k':>2} | {'Silhouette':>10} (Rank) | {'Davies-Bouldin':>13} (Rank) | {'Dunn Index':>10} (Rank) | Rank Final")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['k']:>2} | {row['Silhouette']:>10.3f} ({row['Rank_Silhouette']:>2}) | "
              f"{row['Davies_Bouldin']:>13.3f} ({row['Rank_Davies_Bouldin']:>2}) | "
              f"{row['Dunn_Index']:>10.3f} ({row['Rank_Dunn']:>2}) | {row['Rank_Final']:>8}")
    
    print("-" * 100)
    best_overall = df.loc[df['Rank_Final'] == 1, 'k'].values[0]
    print(f"🏆 MELHOR k PELO RANKING GERAL: {best_overall}")

def main():
    """Função principal do relatório"""
    
    print("📊 ANÁLISE INTERPRETATIVA DAS MÉTRICAS DE CLUSTERING")
    print("🎯 Dataset: Análise Defensiva de Times de Futebol")
    print("📅 Gerado em: Outubro 2025")
    print()
    
    # Gerar relatório interpretativo
    results = create_metrics_interpretation_report()
    
    # Gerar tabela comparativa
    create_comparison_table()
    
    print(f"\n✅ RELATÓRIO CONCLUÍDO")
    print("=" * 50)
    print("📁 Este relatório complementa os gráficos gerados:")
    print("  ├ metricas_clustering_individuais.png")
    print("  ├ metricas_clustering_comparativo.png")
    print("  └ metricas_clustering_normalizadas.png")
    
    print(f"\n🔗 PRÓXIMOS PASSOS SUGERIDOS:")
    print("  ├ Aplicar k={} no clustering final".format(results['consensus_k']))
    print("  ├ Analisar características de cada cluster")
    print("  ├ Validar resultados com especialistas em futebol")
    print("  └ Testar outras técnicas de clustering para comparação")

if __name__ == "__main__":
    main()