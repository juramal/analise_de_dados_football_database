
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Carregar dados
df = pd.read_csv('dataset/transformed_dataset/teamStats_transformed.csv')

# Filtrar colunas relevantes
# ppda: pressão, result: vitória, goals: gols
df['win'] = df['result'].apply(lambda x: 1 if x == 'W' else 0)

# Agrupar por time para médias
team_stats = df.groupby('teamName').agg({
	'ppda': 'mean',
	'win': 'mean',
	'goals': 'mean',
	'shots': 'mean',
	'shotsOnTarget': 'mean',
	'xGoals': 'mean'
}).reset_index()

kmeans = KMeans(n_clusters=6, random_state=42)

# Gráfico de Elbow para escolher o número ideal de clusters
inertia = []

from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# Métricas de qualidade dos clusters
K_range = range(2, 11)  # Métricas só fazem sentido para K >= 2
inertia = []
silhouette_scores = []
davies_bouldin_indices = []
dunn_indices = []

def dunn_index(X, labels):
	clusters = [X[labels == i] for i in np.unique(labels)]
	min_intercluster = np.inf
	max_intracluster = 0
	for i in range(len(clusters)):
		for j in range(i+1, len(clusters)):
			dist = np.min(np.linalg.norm(clusters[i][:,None]-clusters[j], axis=2))
			if dist < min_intercluster:
				min_intercluster = dist
		if len(clusters[i]) > 1:
			intra = np.max(np.linalg.norm(clusters[i][:,None]-clusters[i], axis=2))
			if intra > max_intracluster:
				max_intracluster = intra
	return min_intercluster / max_intracluster if max_intracluster > 0 else 0

for k in K_range:
	km = KMeans(n_clusters=k, random_state=42)
	labels = km.fit_predict(team_stats[['ppda', 'win']])
	inertia.append(km.inertia_)
	sil = silhouette_score(team_stats[['ppda', 'win']], labels)
	db = davies_bouldin_score(team_stats[['ppda', 'win']], labels)
	dunn = dunn_index(team_stats[['ppda', 'win']].values, labels)
	silhouette_scores.append(sil)
	davies_bouldin_indices.append(db)
	dunn_indices.append(dunn)


# Gráfico de Elbow
plt.figure(figsize=(7,4))
plt.plot(K_range, inertia, marker='o')
plt.title('Gráfico de Elbow')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.grid(True)
plt.tight_layout()
plt.savefig('elbow.png')
print('Gráfico de Elbow salvo como elbow.png')

# Gráfico das métricas de qualidade
plt.figure(figsize=(7,4))
plt.plot(K_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.plot(K_range, davies_bouldin_indices, marker='s', label='Davies-Bouldin Index')
plt.plot(K_range, dunn_indices, marker='^', label='Dunn Index')
plt.title('Métricas de Qualidade dos Clusters')
plt.xlabel('Número de Clusters (K)')
plt.legend()
plt.tight_layout()
plt.savefig('cluster_metrics.png')
print('Gráfico de métricas salvo como cluster_metrics.png')


# Clusterização: pressão (ppda baixa = mais pressão), vitórias
X = team_stats[['ppda', 'win']]
kmeans = KMeans(n_clusters=6, random_state=42)
team_stats['cluster'] = kmeans.fit_predict(X)

# Visualização dos clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=team_stats, x='ppda', y='win', hue='cluster', palette='Set1', s=100)
plt.title('Clusters de Times: Pressão vs Vitórias')
plt.xlabel('PPDA (quanto menor, mais pressão)')
plt.ylabel('Proporção de Vitórias')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('clusters.png')
print('Gráfico de clusters salvo como clusters.png')

# Análise de conversão de pressão em gols
print('Médias por cluster:')
cluster_means = team_stats.groupby('cluster')[['ppda','win','goals','xGoals','shots','shotsOnTarget']].mean()
print(cluster_means)

# Mostrar times de cada cluster
for c in sorted(team_stats['cluster'].unique()):
	print(f"\nCluster {c} - Times:")
	print(team_stats[team_stats['cluster'] == c]['teamName'].tolist())

# Destacar clusters mais eficientes (ppda baixo, win alto)
print("\nClusters mais eficientes (ppda baixo, win alto):")
efficiency = cluster_means['win'] / cluster_means['ppda']
top_clusters = efficiency.sort_values(ascending=False).head(2).index.tolist()
for c in top_clusters:
	print(f"Cluster {c}: ppda={cluster_means.loc[c,'ppda']:.2f}, win={cluster_means.loc[c,'win']:.2f}, times={team_stats[team_stats['cluster']==c]['teamName'].tolist()}")
