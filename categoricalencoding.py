import numpy as np
import pandas as pd

df=pd.read_csv("C:\\sampledata\\Customer.csv")
"""print(df.head())
print(df)
print(df.info())
print(df.describe())
"""

print(df['Gender'].value_counts())
print(df['Education'].value_counts())
print(df['Review'].value_counts())
#'Education' and 'Review' both are ordinal categorical variables


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder(categories=[['School','UG','PG'], ['Poor','Average','Good']])
df1_oe=oe.fit_transform(df)
print(type(df1_oe))



#convert aray into dataframe

df1=pd.DataFrame(df1_oe, columns=['Education','Review'])
print(df1.head(2))


#convert into integer
df1['Education']=df1['Education'].astype(int)
df1['Review']=df1['Review'].astype(int)
