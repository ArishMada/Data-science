import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('data1.csv')
init = np.array([[1, 10], [5, 8], [9, 2]])
kmeans = KMeans(n_clusters=3, init=init)
kmeans.fit(data)

identified_clusters = kmeans.fit_predict(data)
print(identified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters

print(data_with_clusters["x"])
plt.scatter(data_with_clusters['x'], data_with_clusters['y'], c='white', marker='o',
            edgecolor='black', s=50)

plt.scatter(data_with_clusters['x'],data_with_clusters['y'],c=data_with_clusters['Clusters'],cmap='rainbow')

plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=50, marker='P',
    c='red', edgecolor='black',
    label='centroids'
)

plt.grid()
plt.show()