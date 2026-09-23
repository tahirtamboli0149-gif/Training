import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\sampledata\\placement.csv")
print(df.head(10))
print(df.shape)
print(df.info())
print(df.describe())


plt.figure(figsize=(10,6))

# First subplot
plt.subplot(1,2,1)
sns.boxplot(x=df['cgpa'], orient='h')
plt.title('Box Plot of CGPA')
plt.xlabel('CGPA')

# Second subplot
plt.subplot(1,2,2)
sns.boxplot(x=df['placement_exam_marks'], orient='h')
plt.title('Box Plot of Placement Exam Marks')
plt.xlabel('Placement Exam Marks')


# Show both together
#plt.show()

print("\nSummary Statistics for CGPA:")
print(df['cgpa'].describe())

print("\nFinding upper and lower limits for CGPA to identify outliers:")
upper_limit=df['cgpa'].mean() + 3 * df['cgpa'].std()
print(f"Upper Limit for CGPA: {upper_limit}")

lower_limit=df['cgpa'].mean() - 3 * df['cgpa'].std()
print(f"Lower Limit for CGPA: {lower_limit}")

print("\nFinding the boundary values for cgpa to identify outliers:")
print(df[(df['cgpa'] > upper_limit) | (df['cgpa'] < lower_limit)])




"""Outlier methods:
1.Trimming: Remove the outliers from the dataset.
2.Capping: Replace the outliers with the nearest boundary value."""

#Removing outliers using trimming method
print("\nRemoving outliers using trimming method:")
df_trimmed = df[(df['cgpa'] <= upper_limit) & (df['cgpa'] >= lower_limit)]
print(df_trimmed)


#Visualizing the data after removing outliers using trimming method
plt.figure(figsize=(10,6))

# First subplot
plt.subplot(1,2,1)
sns.boxplot(x=df['cgpa'], orient='h')
plt.title('Box Plot of CGPA')
plt.xlabel('CGPA')

# Trimmed subplot
plt.subplot(1,2,2)
sns.boxplot(x=df_trimmed['cgpa'], orient='h')
plt.title('Box Plot of CGPA (Trimmed)')
plt.xlabel('CGPA')

#plt.show()


"""IQR method to identify outliers:"""
#elements below Q1-1.5*IQR and above Q3+1.5*IQR are considered outliers
print("\nDescribing the placement_exam_marks column to find outliers using IQR method:")
print(df['placement_exam_marks'].describe())
sns.boxplot(x=df['placement_exam_marks'], orient='h')
plt.title('Box Plot of Placement Exam Marks')
plt.xlabel('Placement Exam Marks')
#plt.show()

Q1 = df['placement_exam_marks'].quantile(0.25)
Q3 = df['placement_exam_marks'].quantile(0.75)
IQR = Q3 - Q1
print("\nQuartiles and IQR for Placement Exam Marks:")
print("value of Q1:",Q1)
print("value of Q3:",Q3)
print("value of IQR:",IQR)

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR
print("Upper Limit for Placement Exam Marks:", upper_limit)
print("Lower Limit for Placement Exam Marks:", lower_limit)

"""FINDING OUTLIESRS"""
print("\n")
print(df[(df['placement_exam_marks'] > upper_limit)])



"""Creating new dataframe using trimming method to remove outliers from placement_exam_marks column:"""

new_df = df[(df['placement_exam_marks'] < upper_limit)]
print(new_df)
print(new_df.shape)



#comparing the boxplots of placement_exam_marks before and after removing outliers using trimming method
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
sns.boxplot(x=df['placement_exam_marks'], orient='h')   
plt.title('Box Plot of Placement Exam Marks')
plt.xlabel('Placement Exam Marks')

plt.subplot(2,2,2)
sns.boxplot(x=new_df['placement_exam_marks'], orient='h')   
plt.title('Box Plot of Placement Exam Marks (Trimmed)')
plt.xlabel('Placement Exam Marks')
#plt.show()


"""SUCCESSFULLY COMPLETED THE 3RD STEP OF EDA- OUTLIER ANALYSIS AND CLEANING"""
