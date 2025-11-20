"""
Classificador Naive Bayes para análise de relação entre jogadores e chutes.
Objetivo: Classificar a probabilidade de um chute resultar em gol com base nas características do jogador e do chute.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class NaiveBayesClassifier:
    def __init__(self, data_dir='dataset/transformed_dataset', output_dir='T2/results'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model = GaussianNB()
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
        
        # Criar variável target: shotResult (Goal/No Goal)
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
        
        # Selecionar features numéricas e encodadas
        feature_cols = [
            'xGoal', 'positionX', 'positionY', 'minute',
            'goals', 'shots_x', 'xGoals_x', 'assists', 'keyPasses',
            'position_encoded', 'shotType_encoded', 'situation_encoded',
            'lastAction_encoded', 'pressure_encoded'
        ]
        
        # Filtrar apenas colunas que existem
        available_features = [col for col in feature_cols if col in df.columns]
        
        self.X = df[available_features].fillna(0)
        self.y = df['target']
        
        print(f"✓ Dataset final: {len(self.X)} amostras")
        print(f"✓ Features: {len(available_features)}")
        print(f"✓ Distribuição target: {self.y.value_counts().to_dict()}")
        
        return df
    
    def train_model(self, test_size=0.3, random_state=42):
        """Treina o modelo Naive Bayes."""
        print("\nTreinando Naive Bayes...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Predições
        self.y_pred = self.model.predict(self.X_test)
        
        print("✓ Modelo treinado")
    
    def evaluate_model(self):
        """Avalia o desempenho do modelo."""
        print("\n" + "="*80)
        print("AVALIAÇÃO DO MODELO - NAIVE BAYES")
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Goal', 'Goal'],
                    yticklabels=['No Goal', 'Goal'])
        plt.title('Confusion Matrix - Naive Bayes')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/naivebayes_confusion_matrix.png', dpi=300)
        print(f"\n✓ Confusion matrix salva em {self.output_dir}/naivebayes_confusion_matrix.png")
        plt.close()
        
        # Feature Importance (via variância das probabilidades)
        self._plot_feature_importance()
        
        # Salvar resultados
        self._save_results(accuracy, cv_scores)
    
    def _plot_feature_importance(self):
        """Plota importância das features baseado nas probabilidades."""
        feature_names = self.X.columns
        
        # Para Naive Bayes, podemos usar a variância das log-probabilidades
        log_probs = self.model.predict_log_proba(self.X_train)
        feature_variance = np.var(log_probs, axis=0)
        
        # Normalizar
        importance = feature_variance / feature_variance.sum()
        
        # Criar DataFrame
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
        plt.xlabel('Importância Relativa')
        plt.title('Top 15 Features - Naive Bayes')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/naivebayes_feature_importance.png', dpi=300)
        print(f"✓ Feature importance salva em {self.output_dir}/naivebayes_feature_importance.png")
        plt.close()
    
    def _save_results(self, accuracy, cv_scores):
        """Salva resultados em CSV."""
        results = {
            'Model': ['Naive Bayes'],
            'Accuracy': [accuracy],
            'CV_Mean': [cv_scores.mean()],
            'CV_Std': [cv_scores.std()],
            'Train_Samples': [len(self.X_train)],
            'Test_Samples': [len(self.X_test)]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{self.output_dir}/naivebayes_results.csv', index=False)
        print(f"✓ Resultados salvos em {self.output_dir}/naivebayes_results.csv")


def main():
    print("="*80)
    print("CLASSIFICAÇÃO: NAIVE BAYES - Relação Jogadores e Chutes")
    print("="*80)
    
    classifier = NaiveBayesClassifier()
    classifier.load_data()
    classifier.prepare_features()
    classifier.train_model()
    classifier.evaluate_model()
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80)


if __name__ == "__main__":
    main()
