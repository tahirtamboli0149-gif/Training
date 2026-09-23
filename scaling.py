import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



df=pd.read_csv('C:\\sampledata\\Feature_Scaling.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n",df.describe())


print("\nAccesesing only Age and Salary columns from the dataset:")
df1=df[['Age','Salary']]
print("\n",df1)
#checking if there are any null values in the dataset
print("\nChecking for null values:")
print(df1.isnull().sum())

"""STARING WITH SCALING METHODS:
1. STANDARD SCALER: Standardization is the process of rescaling the features so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one."""

scaler = StandardScaler()
scaler.fit(df1)
df1_scaled = scaler.transform(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns=df1.columns)
print("\nScaled dataset (Standard Scaler):")
print(df1_scaled)
print("\nMean of scaled features:")
print(scaler.mean_)
print("\nStandard deviation of scaled features:")
print(scaler.scale_)
print("\nDescribing the scaled dataset:")
print(df1_scaled.describe())

df1_scaled=scaler.fit_transform(df1)
print("\nScaled dataset (Standard Scaler) using fit_transform:")
print(df1_scaled)

#now we convert the scaled dataset into a dataframe
df1_scaled=pd.DataFrame(df1_scaled, columns=df1.columns)

#after converting the scaled dataset into a dataframe
print("\nScaled dataset (Standard Scaler) after converting into a dataframe:")
print(df1_scaled)
print("\nDescribing the scaled dataset after converting into a dataframe:")
print(df1_scaled.describe().round(2))
#after roundig2 mean is 0 and sd is 1


"""NOW APPLYING
      2. MINMAX SCALER"""

mc=MinMaxScaler()
mc.fit(df1)
df2_scaled = mc.transform(df1)
df2_scaled = pd.DataFrame(df2_scaled, columns=df1.columns)
mc = MinMaxScaler()
df2_scaled = mc.fit_transform(df1)
df2_scaled = pd.DataFrame(df2_scaled, columns=df1.columns)

print("\nScaled dataset (MinMax Scaler):")
print(df2_scaled)

# Inspect learned parameters
print("\nData min per feature:", mc.data_min_)
print("Data max per feature:", mc.data_max_)
print("Data range per feature:", mc.data_range_)
print("Feature min (target range):", mc.min_)
print("Feature scale (target range):", mc.scale_)

print("\nDescribing the scaled dataset after converting into a dataframe:")
print(df2_scaled.describe().round(2))



