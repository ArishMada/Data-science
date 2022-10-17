import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

kidneyData = pd.read_csv("kidney_disease.csv")

# replacing nominal values with equivalent float
kidneyData["rbc"] = kidneyData["rbc"].replace(to_replace=["normal", "abnormal"],
                                              value=[1, 0])
kidneyData["pc"] = kidneyData["pc"].replace(to_replace=["normal", "abnormal"],
                                            value=[1, 0])
kidneyData["pcc"] = kidneyData["pcc"].replace(to_replace=["present", "notpresent"],
                                              value=[1, 0])
kidneyData["ba"] = kidneyData["ba"].replace(to_replace=["present", "notpresent"],
                                            value=[1, 0])
kidneyData["appet"] = kidneyData["appet"].replace(to_replace=["good", "poor"],
                                                  value=[1, 0])
kidneyData["htn"] = kidneyData["htn"].replace(to_replace=["yes", "no"],
                                              value=[1, 0])
kidneyData["dm"] = kidneyData["dm"].replace(to_replace=["yes", "no"],
                                            value=[1, 0])
kidneyData["cad"] = kidneyData["cad"].replace(to_replace=["yes", "no"],
                                              value=[1, 0])
kidneyData["pe"] = kidneyData["pe"].replace(to_replace=["yes", "no"],
                                            value=[1, 0])
kidneyData["ane"] = kidneyData["ane"].replace(to_replace=["yes", "no"],
                                              value=[1, 0])
kidneyData["classification"] = kidneyData["classification"].replace(to_replace=["ckd", "notckd"],
                                                                    value=[1, 0])

# cleaning the rows that have empty cells
kidneyData = kidneyData.dropna()

x = kidneyData.drop("classification", axis=1).loc[:, ["age", "hemo"]]
y = kidneyData['classification'].astype("int")

# print(microbeDataSample["microorganisms"].unique())
print(kidneyData)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy:{metrics.accuracy_score(y_test, y_pred) * 100}%')

# plotting with only points
# plt.scatter(kidneyData["age"][kidneyData["classification"] == 1], kidneyData["hemo"][kidneyData["classification"] == 1],
#             color='red', marker='o', label='kidney disease')
# plt.scatter(kidneyData["age"][kidneyData["classification"] == 0], kidneyData["hemo"][kidneyData["classification"] == 0],
#             color='blue', marker='*', label='no kidney disease')
# plt.xlim()
# plt.xlabel('age')
# plt.ylabel("hemoglobin")
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
title = 'Kidney disease classification'
# Set-up grid for plotting.
X0, X1 = x.loc[:, "age"], x.loc[:, "hemo"]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap="Set1", alpha=0.8)

plt.scatter(kidneyData["age"][kidneyData["classification"] == 1], kidneyData["hemo"][kidneyData["classification"] == 1],
            color='black', marker='o', label='kidney disease')
plt.scatter(kidneyData["age"][kidneyData["classification"] == 0], kidneyData["hemo"][kidneyData["classification"] == 0],
            color='blue', marker='*', label='no kidney disease')

plt.xlim()
ax.set_ylabel('hemoglobin number')
ax.set_xlabel('age')
plt.locator_params('x', nbins=20)
ax.locator_params('y', nbins=10)
ax.set_title(title)
ax.legend()
plt.show()
