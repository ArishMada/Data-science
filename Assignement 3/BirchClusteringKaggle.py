from sklearn.cluster import Birch
from matplotlib import pyplot as plt
import pandas as pd

# define dataset
X = pd.read_csv('wine-clustering.csv')
# define the model
model = Birch(threshold=0.01, n_clusters=3)
# fit the model
model.fit(X)
# assign a cluster to each example
identified_clusters = model.predict(X)
data_with_clusters = X.copy()
data_with_clusters['Clusters'] = identified_clusters

plt.scatter(data_with_clusters['x'], data_with_clusters['y'], c='white', marker='o',
            edgecolor='black', s=50)

plt.scatter(data_with_clusters['x'],data_with_clusters['y'], c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()