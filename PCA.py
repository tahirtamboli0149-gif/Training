import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# reading dataset
df = pd.read_csv('C:\\sampledata\\load_digits.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print("Missing values:", df.isnull().sum().sum())



x=df.drop('target',axis=1)
y=df['target']
print("Shape of x:",x.shape)
print("Shape of y:",y.shape)


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print("\n",x_scaled)


#splitting data into train and test part
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=30)
print("\nSplitting data")
print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


#Applying Logistic Regression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

#test score on testing data
print("\nScore of tested data")
print(model.score(x_test, y_test))


#Using PCA to reduce dimentions
pca= PCA(0.95)
x_pca=pca.fit_transform(x)
print("\nShape of PCA")
print(x_pca.shape)


x_train_pca,x_test_pca,y_train_pca,y_test_pca=train_test_split(x_pca,y,test_size=0.2,random_state=30)


model=LogisticRegression()
model.fit(x_train_pca,y_train)
model.score(x_test_pca,y_test)
print("\nScore of tested data")
print(model.score(x_test_pca, y_test))
#we take 29 input features in this example

