import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('wine-clustering.csv')
x = data.iloc[:, 6:9:2]

print(x)

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)
#
data_with_clusters = x
data_with_clusters['Clusters'] = identified_clusters

plt.scatter(x['Flavanoids'], x['Proanthocyanins'], c='white', marker='o',
            edgecolor='black', s=50)

plt.scatter(x['Flavanoids'],x['Proanthocyanins'],c=data_with_clusters['Clusters'],cmap='rainbow')

plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=50, marker='P',
    c='red', edgecolor='black',
    label='centroids'
)

plt.show()