import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# reading dataset
df = pd.read_csv('C:\\sampledata\\heart.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print("Shape of dataset:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.info())



# checking for outliers using IQR method
print("\nOUTLIER CHECK:")
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Define outliers for this column
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"{column}:")
    print("  Lower bound:", lower_bound)
    print("  Upper bound:", upper_bound)
    print("  Number of outliers:", len(outliers))


# removing outliers
original_rows = len(df)
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

removed_rows = original_rows - len(df)
print("\nOUTLIER REMOVAL:")
print("Original rows:", original_rows)
print("Rows removed:", removed_rows)
print("Remaining rows:", len(df))

#splitting data into train test part
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print("\nShape of x:", x.shape)
print("Shape of y:", y.shape)
print("\nFeatures used:")
print(x.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# splitting data into train and test part
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#preprocessing pipeline
processor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ]  
)    

#logistic regression model
print("\nPreprocessing pipeline created successfully!")

# logistic regression model
logistic_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred_log = logistic_model.predict(X_test)

print("\nLOGISTIC MODEL RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))


#decision tree model
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)
print("\nDECISION TREE RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))



#random forest model
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)
random_forest_model.fit(X_train, y_train)
y_pred_forest = random_forest_model.predict(X_test)
print("\nRANDOM FOREST RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_forest))


#knn model
from sklearn.neighbors import KNeighborsClassifier
knn_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]
)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("\nKNN RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))


#svm model
svm_model = Pipeline(
    steps=[
        ("preprocessor", processor),
        ("classifier", SVC(random_state=42))
    ]
)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nSVM RESULTS:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))




print("\nSince the Random Forest model has the highest accuracy, we will use it for further evaluation metrics.")
print("Precision:", precision_score(y_test, y_pred_forest, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_forest, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_forest, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_forest))


