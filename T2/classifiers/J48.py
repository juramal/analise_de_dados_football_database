"""
Classificador J48 (Decision Tree C4.5) para análise de relação entre jogadores e chutes.
Objetivo: Classificar a probabilidade de um chute resultar em gol usando árvores de decisão.
Nota: Implementação usando DecisionTreeClassifier do sklearn com parâmetros similares ao J48.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class J48Classifier:
    def __init__(self, data_dir='dataset/transformed_dataset', output_dir='T2/results'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # Parâmetros similares ao J48 do Weka
        self.model = DecisionTreeClassifier(
            criterion='entropy',  # J48 usa ganho de informação
            splitter='best',
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.label_encoders = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Carrega todos os datasets transformados."""
        print("Carregando datasets...")
        
        self.shots_df = pd.read_csv(f'{self.data_dir}/shotStats_transformed.csv')
        self.players_df = pd.read_csv(f'{self.data_dir}/playerStatsinGame_transformed.csv')
        self.teams_df = pd.read_csv(f'{self.data_dir}/teamStats_transformed.csv')
        self.games_df = pd.read_csv(f'{self.data_dir}/gameStats_transformed.csv')
        
        print(f"✓ Shots: {len(self.shots_df)} registros")
        print(f"✓ Players: {len(self.players_df)} registros")
        print(f"✓ Teams: {len(self.teams_df)} registros")
        print(f"✓ Games: {len(self.games_df)} registros")
    
    def prepare_features(self):
        """Prepara features unindo todos os datasets."""
        print("\nPreparando features...")
        
        # Juntar shots com players
        df = self.shots_df.merge(
            self.players_df,
            left_on=['gameID', 'shooterName'],
            right_on=['gameID', 'playerName'],
            how='left'
        )
        
        # Juntar com teams
        df = df.merge(
            self.teams_df[['gameID', 'teamName', 'xGoals', 'shots', 'shotsOnTarget', 'pressure']],
            on='gameID',
            how='left',
            suffixes=('', '_team')
        )
        
        # Juntar com games
        df = df.merge(
            self.games_df[['gameID', 'leagueName', 'season']],
            on='gameID',
            how='left'
        )
        
        # Criar variável target
        df['target'] = df['shotResult'].apply(lambda x: 1 if x == 'Goal' else 0)
        
        # Remover linhas com valores nulos críticos
        df = df.dropna(subset=['target', 'xGoal', 'position', 'shotType', 'situation'])
        
        # Encoding de variáveis categóricas
        categorical_cols = ['position', 'shotType', 'situation', 'lastAction', 'pressure']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        # Selecionar features
        feature_cols = [
            'xGoal', 'positionX', 'positionY', 'minute',
            'goals', 'shots_x', 'xGoals_x', 'assists', 'keyPasses',
            'position_encoded', 'shotType_encoded', 'situation_encoded',
            'lastAction_encoded', 'pressure_encoded'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        self.X = df[available_features].fillna(0)
        self.y = df['target']
        self.feature_names = available_features
        
        print(f"✓ Dataset final: {len(self.X)} amostras")
        print(f"✓ Features: {len(available_features)}")
        print(f"✓ Distribuição target: {self.y.value_counts().to_dict()}")
        
        return df
    
    def train_model(self, test_size=0.3, random_state=42):
        """Treina o modelo J48."""
        print("\nTreinando J48 (Decision Tree)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Predições
        self.y_pred = self.model.predict(self.X_test)
        
        print("✓ Modelo treinado")
        print(f"✓ Profundidade da árvore: {self.model.get_depth()}")
        print(f"✓ Número de folhas: {self.model.get_n_leaves()}")
    
    def evaluate_model(self):
        """Avalia o desempenho do modelo."""
        print("\n" + "="*80)
        print("AVALIAÇÃO DO MODELO - J48 (Decision Tree)")
        print("="*80)
        
        # Acurácia
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"\nAcurácia: {accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                    target_names=['No Goal', 'Goal']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=['No Goal', 'Goal'],
                    yticklabels=['No Goal', 'Goal'])
        plt.title('Confusion Matrix - J48')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/j48_confusion_matrix.png', dpi=300)
        print(f"\n✓ Confusion matrix salva em {self.output_dir}/j48_confusion_matrix.png")
        plt.close()
        
        # Feature Importance
        self._plot_feature_importance()
        
        # Visualizar árvore (versão simplificada)
        self._plot_tree()
        
        # Salvar resultados
        self._save_results(accuracy, cv_scores)
    
    def _plot_feature_importance(self):
        """Plota importância das features."""
        importance = self.model.feature_importances_
        
        feat_imp_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='purple', alpha=0.7)
        plt.xlabel('Importância')
        plt.title('Top 15 Features - J48')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/j48_feature_importance.png', dpi=300)
        print(f"✓ Feature importance salva em {self.output_dir}/j48_feature_importance.png")
        plt.close()
    
    def _plot_tree(self):
        """Visualiza a árvore de decisão (primeiros 3 níveis)."""
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                  max_depth=3,
                  feature_names=self.feature_names,
                  class_names=['No Goal', 'Goal'],
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title('Árvore de Decisão J48 (Primeiros 3 níveis)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/j48_tree_visualization.png', dpi=300, bbox_inches='tight')
        print(f"✓ Árvore visualizada em {self.output_dir}/j48_tree_visualization.png")
        plt.close()
    
    def _save_results(self, accuracy, cv_scores):
        """Salva resultados em CSV."""
        results = {
            'Model': ['J48 (Decision Tree)'],
            'Accuracy': [accuracy],
            'CV_Mean': [cv_scores.mean()],
            'CV_Std': [cv_scores.std()],
            'Train_Samples': [len(self.X_train)],
            'Test_Samples': [len(self.X_test)],
            'Tree_Depth': [self.model.get_depth()],
            'Tree_Leaves': [self.model.get_n_leaves()]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.output_dir}/j48_results.csv', index=False)
        print(f"✓ Resultados salvos em {self.output_dir}/j48_results.csv")


def main():
    print("="*80)
    print("CLASSIFICAÇÃO: J48 (Decision Tree) - Relação Jogadores e Chutes")
    print("="*80)
    
    classifier = J48Classifier()
    classifier.load_data()
    classifier.prepare_features()
    classifier.train_model()
    classifier.evaluate_model()
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80)


if __name__ == "__main__":
    main()
