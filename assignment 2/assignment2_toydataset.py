from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data)
df["label"] = iris.target
# print(iris.target)
# for i in range(149):
#     if df.loc
print(iris.data)
# print the names of the 13 features
print("Features: ", iris.feature_names)

print("Labels: ", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(iris.data[:, :2], iris.target, test_size=0.2, random_state=0)

clf = svm.SVC(kernel='rbf', gamma=0.6, C=0.1)  # used rbf as it seemed more accurate than polynomial

clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print(f'Accuracy:{round(metrics.accuracy_score(y_test, y_pred) * 100, 2)}%')

# plotting
plt.scatter(df[0][df["label"] == 0], df[1][df["label"] == 0],
            color='red', marker='o', label='setosa')
plt.scatter(df[0][df["label"] == 1], df[1][df["label"] == 1],
            color='blue', marker='*', label='versicolor')
plt.scatter(df[0][df["label"] == 2], df[1][df["label"] == 2],
            color='black', marker='^', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='upper left')
plt.show()