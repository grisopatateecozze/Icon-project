"""
@author: Grisolia Giuseppe

Modulo che analizza il dataset con l'algoritmo di apprendimento non supervisionato K-MEANS.
Si sceglie il numero ottimale di clusters. Si esegue il k-mean con quel numero di clusters
e si calcolano e valutano la somma dei quadrati intra-cluster (WCSS) e la Silhouette Score.
Infine viene creato un nuovo file .csv con il dataset aggiornato con una colonna clusters
dove ci sono le varie classificazioni di ogni paziente nei tre cluster.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset0 = pd.read_csv('../Ontologia/TCGA_InfoWithGrade.csv')
dataset = dataset0

# esecuzione K-means per un range di K cluster
dataset = np.array(dataset)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=40)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)  

# tracciamento del grafico
plt.plot(range(1, 11), wcss, 'bx-')
plt.title('elbow method')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('WCSS')
plt.show() 

kmeans = KMeans(n_clusters=3, n_init=10, random_state=40)
kmeans.fit(dataset)

# Calcolo della somma dei quadrati intra-cluster (WCSS)
wcss = kmeans.inertia_
print("WCSS:", wcss)

# Calcolo dello Silhouette Score
silhouette_avg = silhouette_score(dataset, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)
cluster_result = kmeans.labels_

dataset0['cluster'] = cluster_result

# Riordina le colonne del DataFrame
columns_order = list(dataset0.columns)
columns_order.remove('Grade')
columns_order.append('Grade')
columns_order.remove('cluster')
columns_order.insert(-1, 'cluster')
dataset_reordered = dataset0[columns_order]

dataset_reordered.to_csv('TCGA-clusters.csv', index=False)
