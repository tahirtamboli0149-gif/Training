import numpy as np
import pandas as pd

df=pd.read_csv("C:\\sampledata\\EDA_data.csv")
print(df)



#checking missing values
print(df.info())
print(df.isnull().sum())
print(df.isnull().sum().sum())




#seperating numerical and categorical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
#categorical_columns = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", numerical_columns)
#print("Categorical Columns:", categorical_columns)


print("Missing values in numerical columns:")
print(df[numerical_columns].isnull().sum())