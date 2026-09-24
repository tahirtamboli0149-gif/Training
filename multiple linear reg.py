import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#importing model
from sklearn.linear_model import LinearRegression
#imporing evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#reading dataset
df = pd.read_csv('C:\\sampledata\\house price prediction.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print(df.isnull().sum().sum())
print(df.info())   


#'price' column is target variable
x = df.drop('price', axis=1)
y = df['price']
print("shape of x:", x.shape)
print("shape of y:", y.shape)


#splitting data into train test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)
print("\nSplitting data")
print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


#Training dataset using linear regression
lr = LinearRegression()

#applying model 
lr.fit(x_train, y_train)

#test score on testing data
print("\nScore of tested data")
print(lr.score(x_test, y_test))


"""CALCULATING THE SLOPE USING Y=mx+b"""
#slope(m) coef_
m = lr.coef_
print(m)

#intercept
b = lr.intercept_
print(b)


print("\n", x_test.head())
print("\n", y_test.head())


#predict all x_tests
y_pred = lr.predict(x_test)
print("\n", y_pred)


"""linear model evaluation metrics
    1. loss function
    2. cost function
    3. MAE
    4. MSE
    5. RMSE
    6. R2 Score"""


#MAE
print("\nMAE", mean_absolute_error(y_test, y_pred))

#MSE
print("\nMSE", mean_squared_error(y_test, y_pred))

#RMSE
print("\nRMSE", np.sqrt(mean_squared_error(y_test, y_pred)))

#R2_Score
print("\nR2_Score", r2_score(y_test, y_pred))
