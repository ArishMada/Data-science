from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

# Load dataset
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data)
df["label"] = iris.target

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


# plotting in 2D without surface color
# plt.scatter(df[0][df["label"] == 0], df[1][df["label"] == 0],
#             color='red', marker='o', label='setosa')
# plt.scatter(df[0][df["label"] == 1], df[1][df["label"] == 1],
#             color='blue', marker='*', label='versicolor')
# plt.scatter(df[0][df["label"] == 2], df[1][df["label"] == 2],
#             color='black', marker='^', label='virginica')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc='upper left')
# plt.show()

# plotting with surface color
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


fig, ax = plt.subplots()
# title for the plots
title = ('Types of iris based on sepal length and width')
# Set-up grid for plotting.
X0, X1 = iris.data[:, 0], iris.data[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(df[0][df["label"] == 0], df[1][df["label"] == 0],
            color='white', marker='o', label='setosa')
plt.scatter(df[0][df["label"] == 1], df[1][df["label"] == 1],
            color='blue', marker='*', label='versicolor')
plt.scatter(df[0][df["label"] == 2], df[1][df["label"] == 2],
            color='black', marker='^', label='virginica')

ax.set_ylabel('sepal length(cm)')
ax.set_xlabel('sepal width(cm)')
plt.locator_params('x', nbins=20)
ax.locator_params('y', nbins=10)
ax.set_title(title)
ax.legend()
plt.show()
