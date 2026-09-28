"""K-nearest neighbors algorithm module."""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#reading dataset
df = pd.read_csv('C:\\sampledata\\breast_cancer.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print(df.isnull().sum().sum())
print(df.info())   


#defining input and output
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print("Shape of x", x.shape)
print("Shape of y", y.shape)


#splitting data into train test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=51)
print("\nSplitting data")
print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


#applying KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

#checking score
print("\nKNN score of tested data")
print(knn.score(x_test, y_test))
