import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Heart.csv")

df.info()

df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.head(5))

print(df.tail(5))

print(df.describe())

print(df.isna().sum().sum())  # all dataset

df.dropna(inplace=True)

print(df.shape[0])  # this return 297 which is the number of rows left after dropping the one with missing values

corr = df.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
# plt.show()

print(f'variance for age: {df.var()["Age"]}')

df["ChestPain"] = df["ChestPain"].replace(to_replace=["typical", "asymptomatic", "nonanginal", "nontypical"],
                                              value=[0, 1, 2, 3])
df["Thal"] = df["ChestPain"].replace(to_replace=["fixed", "normal", "reversable"],
                                              value=[0, 1, 2])
df["AHD"] = df["AHD"].replace(to_replace=["No", "Yes"],
                                              value=[0, 1])
# print(df)
