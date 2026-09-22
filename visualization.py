import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


#visualizing missing values using heatmap
df=pd.read_csv("C:\\sampledata\\EDA_data.csv")
print(sns.heatmap(df.isnull()))
plt.show()

print(df.describe())


#checking the mean of data using subplots


sns.displot(df['Age'], kde=True)
plt.show()

sns.histplot(df['EstimatedSalary'], kde=True)
plt.show()


sns.histplot(df['Purchased'], kde=True)
plt.show()


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(1,2,2)
sns.histplot(df['EstimatedSalary'], kde=True)
plt.title('Estimated Salary Distribution')

"""plt.subplot(2,2,1)
sns.histplot(df['Purchased'], kde=True)
plt.title('Purchased Distribution')"""


plt.tight_layout()
plt.show()



#filling missing values with mean and median of the data

df['Age']=df['Age'].fillna(df['Age'].mean())

df['EstimatedSalary']=df['EstimatedSalary'].fillna(df['EstimatedSalary'].mean())


df['Purchased']=df['Purchased'].fillna(df['Purchased'].median())          
print(df.isnull().sum())    

#filling catrgorical data with mode of the data

df1=df.select_dtypes(include=['str']).columns
print(df1)




df.to_csv("25sept.csv")
import os
os.getcwd()