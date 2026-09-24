import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# importing models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# importing evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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


# applying Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# checking score
print("\nLinear Regression R2 Score (accuracy of regression):")
print(lr.score(x_test, y_test))

# slope (m) and intercept (b)
print("\nCoefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

# predict all x_tests
y_pred = lr.predict(x_test)
print("\nPredictions:", y_pred)

# evaluation metrics for regression
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# applying Decision Tree model
dc = DecisionTreeClassifier(criterion='gini', max_depth=3)
dc.fit(x_train, y_train)
print("\nDecision Tree score of tested data")
print(dc.score(x_test, y_test))


# applying Random Forest model
rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=51)
rf.fit(x_train, y_train)
print("\nRandom Forest score of tested data")
print(rf.score(x_test, y_test))


# evaluation matrix for classification model (Decision Tree)
y_pre = dc.predict(x_test)
cm = confusion_matrix(y_test, y_pre)
print("\nConfusion Matrix:\n", cm)

# visualising the confusion matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# classification metrics (only meaningful if y is categorical)
print("\nAccuracy:", accuracy_score(y_test, y_pre))
print("Precision:", precision_score(y_test, y_pre, zero_division=0))
print("Recall:", recall_score(y_test, y_pre, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pre, zero_division=0))


# applying KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
print("\nKNN score of tested data")
print(knn.score(x_test, y_test))
