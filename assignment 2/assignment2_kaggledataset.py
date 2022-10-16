import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

kidneyData = pd.read_csv("kidney_disease.csv")

kidneyData.dropna(subset=["sc"], inplace=True)
kidneyData.dropna(subset=["htn"], inplace=True)

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

x = kidneyData.drop("classification", axis=1)
y = kidneyData['classification'].astype("int")

# print(microbeDataSample["microorganisms"].unique())
print(kidneyData)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy:{metrics.accuracy_score(y_test, y_pred) * 100}%')

plt.scatter(kidneyData["age"][kidneyData["classification"] == 1], kidneyData["hemo"][kidneyData["classification"] == 1],
            color='red', marker='o', label='kidney disease')
plt.scatter(kidneyData["age"][kidneyData["classification"] == 0], kidneyData["hemo"][kidneyData["classification"] == 0],
            color='blue', marker='*', label='no kidney disease')
plt.xlim()
plt.xlabel('age')
plt.ylabel("hemoglobin")
plt.legend(loc='upper left')
plt.show()
