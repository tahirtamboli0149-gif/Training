import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# reading dataset
df = pd.read_csv('C:\\sampledata\\uber.csv')

print(df.head(10))

print("\nDescribing the dataset:")
print("\n", df.describe())

print("Shape of dataset:", df.shape)

print("Missing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print(df.info())


# removing missing values
print("\nREMOVING MISSING VALUES:")

original_rows = len(df)

df = df.dropna()

removed_rows = original_rows - len(df)

print("Original rows:", original_rows)
print("Rows removed:", removed_rows)
print("Remaining rows:", len(df))


# removing unnecessary columns
print("\nREMOVING UNNECESSARY COLUMNS:")

df = df.drop(["Unnamed: 0", "key"], axis=1)

print("Columns after removal:")
print(df.columns.tolist())


# converting pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


# extracting useful information from pickup_datetime
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year


# removing original pickup_datetime column
df = df.drop("pickup_datetime", axis=1)


print("\nDataset after feature extraction:")
print(df.head())


# checking for outliers using IQR method
print("\nOUTLIER CHECK:")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("Numeric columns:", numeric_columns.tolist())

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Define outliers for this column
    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"{column}:")
    print("  Lower bound:", lower_bound)
    print("  Upper bound:", upper_bound)
    print("  Number of outliers:", len(outliers))


# visualizing the outliers using boxplot
numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

plt.figure(figsize=(15, 8))

for i, col in enumerate(numeric_cols, 1):

    plt.subplot(
        len(numeric_cols)//3 + 1,
        3,
        i
    )

    sns.boxplot(x=df[col])

    plt.title(col)

plt.tight_layout()
plt.show()


# visualizing pickup locations
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=df["pickup_longitude"],
    y=df["pickup_latitude"],
    alpha=0.3
)

plt.title("Pickup Locations")
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")

plt.show()


# visualizing dropoff locations
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=df["dropoff_longitude"],
    y=df["dropoff_latitude"],
    alpha=0.3
)

plt.title("Dropoff Locations")
plt.xlabel("Dropoff Longitude")
plt.ylabel("Dropoff Latitude")

plt.show()


# removing invalid values
print("\nREMOVING INVALID VALUES:")

original_rows = len(df)

df = df[df["fare_amount"] > 0]

df = df[df["passenger_count"] > 0]

df = df[
    (df["pickup_latitude"] >= -90) &
    (df["pickup_latitude"] <= 90)
]

df = df[
    (df["dropoff_latitude"] >= -90) &
    (df["dropoff_latitude"] <= 90)
]

df = df[
    (df["pickup_longitude"] >= -180) &
    (df["pickup_longitude"] <= 180)
]

df = df[
    (df["dropoff_longitude"] >= -180) &
    (df["dropoff_longitude"] <= 180)
]

removed_rows = original_rows - len(df)

print("Original rows:", original_rows)
print("Rows removed:", removed_rows)
print("Remaining rows:", len(df))


# selecting features and target for model training

x = df.drop("fare_amount", axis=1)

y = df["fare_amount"]

print("\nShape of x:", x.shape)

print("Shape of y:", y.shape)

print("\nFeatures used:")

print(x.columns.tolist())

print("\nTarget variable:")
print("fare_amount")

print("\nTarget statistics:")
print(y.describe())


# splitting data into train and test part

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData split successfully!")

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))


# selecting numeric columns

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumeric features:")

print(numeric_columns.tolist())


# preprocessing pipeline

processor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns)
    ]
)

print("\nPreprocessing pipeline created successfully!")


# linear regression model

linear_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("regressor", LinearRegression())
    ]
)

print("\nTraining Linear Regression...")

linear_model.fit(X_train, y_train)

print("Model trained successfully!")


# predictions using linear regression

y_pred_linear = linear_model.predict(X_test)


print("\nLINEAR REGRESSION RESULTS:")

print("MAE:",
      mean_absolute_error(y_test, y_pred_linear))

print("MSE:",
      mean_squared_error(y_test, y_pred_linear))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, y_pred_linear)))

print("R2 Score:",
      r2_score(y_test, y_pred_linear))


# decision tree model

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("regressor", DecisionTreeRegressor(
            random_state=42
        ))
    ]
)

print("\nTraining Decision Tree Regressor...")

decision_tree_model.fit(X_train, y_train)

print("Model trained successfully!")


# predictions using decision tree

y_pred_tree = decision_tree_model.predict(X_test)


print("\nDECISION TREE RESULTS:")

print("MAE:",
      mean_absolute_error(y_test, y_pred_tree))

print("MSE:",
      mean_squared_error(y_test, y_pred_tree))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, y_pred_tree)))

print("R2 Score:",
      r2_score(y_test, y_pred_tree))


# random forest model

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

print("\nTraining Random Forest Regressor...")

random_forest_model.fit(X_train, y_train)

print("Model trained successfully!")


# predictions using random forest

y_pred_forest = random_forest_model.predict(X_test)


print("\nRANDOM FOREST RESULTS:")

print("MAE:",
      mean_absolute_error(y_test, y_pred_forest))

print("MSE:",
      mean_squared_error(y_test, y_pred_forest))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, y_pred_forest)))

print("R2 Score:",
      r2_score(y_test, y_pred_forest))


# comparing all models

print("\nMODEL COMPARISON:")

linear_r2 = r2_score(y_test, y_pred_linear)

tree_r2 = r2_score(y_test, y_pred_tree)

forest_r2 = r2_score(y_test, y_pred_forest)

print("\nLinear Regression R2 Score:",
      linear_r2)

print("Decision Tree R2 Score:",
      tree_r2)

print("Random Forest R2 Score:",
      forest_r2)


# final model evaluation

print("\nFINAL RANDOM FOREST MODEL EVALUATION:")

print("MAE:",
      mean_absolute_error(y_test, y_pred_forest))

print("MSE:",
      mean_squared_error(y_test, y_pred_forest))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, y_pred_forest)))

print("R2 Score:",
      r2_score(y_test, y_pred_forest))


# actual vs predicted values

comparison = pd.DataFrame({
    "Actual Fare": y_test.values,
    "Predicted Fare": y_pred_forest
})

print("\nACTUAL VS PREDICTED FARES:")

print(comparison.head(10))


# visualizing actual vs predicted values

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred_forest,
    alpha=0.3
)

plt.xlabel("Actual Fare")

plt.ylabel("Predicted Fare")

plt.title("Actual vs Predicted Fare")

plt.show()