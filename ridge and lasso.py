import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# importing models
from sklearn.linear_model import LinearRegression
# importing evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso



# reading dataset
df = pd.read_csv('C:\\sampledata\\house price prediction.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.info())


# defining input and output
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print("Shape of x", x.shape)
print("Shape of y", y.shape)


# splitting data into train test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=51)
print("\nSplitting data")
print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


sc = StandardScaler()
sc.fit(x_train)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# applying Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# checking score
print("\nLinear Regression R2 Score (accuracy of regression):")
print(lr.score(x_test, y_test))

# predict all x_tests
y_pred = lr.predict(x_test)
print("\nPredictions:", y_pred)

# evaluation metrics for regression
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


rd = Ridge(alpha =2)
rd.fit(x_train,y_train)
print(rd.score(x_test,y_test))


ls = Lasso(alpha = 3)
ls.fit(x_train,y_train)
print(ls.score(x_test,y_test))