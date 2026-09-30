import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# reading dataset
df = pd.read_csv('C:\\sampledata\\house price prediction.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print("Shape of dataset:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.info())

# defining input and output
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print("\nShape of x:", x.shape)
print("Shape of y:", y.shape)

# splitting data into train test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=51)
print("\nSplitting data")
print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# scaling features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""
print("\nScaled x_train:\n", x_train)
print("\nScaled x_test:\n", x_test)"""

# applying SVM model (RBF kernel)
svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(x_train, y_train)

# checking score
print("\nSVR R2 Score (accuracy of regression):")
print(svr_rbf.score(x_test, y_test))

# predictions
y_pred = svr_rbf.predict(x_test)
print("\nPredictions:\n", y_pred)

# evaluation metrics
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
