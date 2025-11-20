"""
Script Principal - Execução de Todos os Classificadores
========================================================

Este script executa todos os classificadores e compara seus resultados.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Importar os classificadores
sys.path.append('T2/classifiers')

def run_all_classifiers():
    """Executa todos os classificadores."""
    
    print("="*80)
    print("EXECUÇÃO DE TODOS OS CLASSIFICADORES")
    print("="*80)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    classifiers = [
        ('NaiveBayes', 'T2/classifiers/NaiveBayes.py'),
        ('J48', 'T2/classifiers/J48.py'),
        ('RandomForest', 'T2/classifiers/RandomForest.py'),
        ('MultilayerPerceptron', 'T2/classifiers/MultilayerPerceptron.py')
    ]
    
    for name, script_path in classifiers:
        print("\n" + "="*80)
        print(f"Executando: {name}")
        print("="*80)
        
        try:
            os.system(f'python {script_path}')
            print(f"\n✓ {name} concluído com sucesso")
        except Exception as e:
            print(f"\n✗ Erro ao executar {name}: {str(e)}")
    
    print("\n" + "="*80)
    print("TODOS OS CLASSIFICADORES EXECUTADOS")
    print("="*80)


def compare_results():
    """Compara os resultados de todos os classificadores."""
    
    print("\n" + "="*80)
    print("COMPARAÇÃO DE RESULTADOS")
    print("="*80)
    
    results_dir = 'T2/results'
    
    # Lista de arquivos de resultados
    result_files = [
        'naivebayes_results.csv',
        'j48_results.csv',
        'randomforest_results.csv',
        'mlp_results.csv'
    ]
    
    all_results = []
    
    for file in result_files:
        file_path = f'{results_dir}/{file}'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_results.append(df)
    
    if all_results:
        # Combinar todos os resultados
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Exibir tabela comparativa
        print("\nTabela Comparativa:")
        print(combined_results.to_string(index=False))
        
        # Salvar comparação
        combined_results.to_csv(f'{results_dir}/comparison_all_classifiers.csv', index=False)
        print(f"\n✓ Comparação salva em {results_dir}/comparison_all_classifiers.csv")
        
        # Gráfico comparativo de acurácia
        plot_comparison(combined_results, results_dir)
    else:
        print("Nenhum resultado encontrado para comparação.")


def plot_comparison(df, output_dir):
    """Cria gráfico comparativo dos modelos."""
    
    if 'Accuracy' in df.columns and 'Model' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Gráfico de barras - Acurácia
        plt.subplot(1, 2, 1)
        colors = ['#4472C4', '#8B3A8B', '#FF8C00', '#2E8B57']
        bars = plt.bar(df['Model'], df['Accuracy'], color=colors, alpha=0.7)
        plt.ylabel('Acurácia')
        plt.title('Comparação de Acurácia entre Modelos')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Gráfico de barras - Cross-Validation
        if 'CV_Mean' in df.columns:
            plt.subplot(1, 2, 2)
            bars = plt.bar(df['Model'], df['CV_Mean'], 
                          yerr=df.get('CV_Std', 0), 
                          color=colors, alpha=0.7, capsize=5)
            plt.ylabel('CV Score')
            plt.title('Comparação de Cross-Validation')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_chart.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico comparativo salvo em {output_dir}/comparison_chart.png")
        plt.close()


def generate_summary_report():
    """Gera relatório resumido da análise."""
    
    results_dir = 'T2/results'
    
    print("\n" + "="*80)
    print("RELATÓRIO RESUMIDO")
    print("="*80)
    
    comparison_file = f'{results_dir}/comparison_all_classifiers.csv'
    
    if os.path.exists(comparison_file):
        df = pd.read_csv(comparison_file)
        
        print("\n📊 Melhores Modelos por Métrica:\n")
        
        # Melhor acurácia
        if 'Accuracy' in df.columns:
            best_acc = df.loc[df['Accuracy'].idxmax()]
            print(f"🏆 Melhor Acurácia: {best_acc['Model']} ({best_acc['Accuracy']:.4f})")
        
        # Melhor CV
        if 'CV_Mean' in df.columns:
            best_cv = df.loc[df['CV_Mean'].idxmax()]
            print(f"🏆 Melhor Cross-Validation: {best_cv['Model']} ({best_cv['CV_Mean']:.4f})")
        
        # Melhor AUC (se disponível)
        if 'AUC' in df.columns:
            df_with_auc = df.dropna(subset=['AUC'])
            if not df_with_auc.empty:
                best_auc = df_with_auc.loc[df_with_auc['AUC'].idxmax()]
                print(f"🏆 Melhor AUC: {best_auc['Model']} ({best_auc['AUC']:.4f})")
        
        print("\n" + "="*80)


def main():
    """Função principal."""
    
    # Criar diretório de resultados
    os.makedirs('T2/results', exist_ok=True)
    
    # Executar todos os classificadores
    run_all_classifiers()
    
    # Comparar resultados
    compare_results()
    
    # Gerar relatório
    generate_summary_report()
    
    print(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ Análise completa finalizada!")
    print("✓ Todos os resultados estão em: T2/results/")


if __name__ == "__main__":
    main()
