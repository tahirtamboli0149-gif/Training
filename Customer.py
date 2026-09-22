import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

print("DataFrame with CSV file............\n")

# Fix file path spacing
df = pd.read_csv("C:\\sampledata\\Customer.csv")
print(df.head(10))
print(df.info())

# Value counts
print(df['Gender'].value_counts())
print(df['Purchased'].value_counts())
print(df['Review'].value_counts())
print(df['Education'].value_counts())

# Ordinal Encoding for Education & Review
df1 = df[['Education','Review']]  # safer than iloc
print(df1.head(10))

oe = OrdinalEncoder(categories=[['School','UG','PG'],['Poor','Average','Good']])
df1_oe = oe.fit_transform(df1)
print(df1_oe)

df1 = pd.DataFrame(df1_oe, columns=['Education','Review'])
df1 = df1.astype('int64')
print(df1.head(10))
print(df1.info())

# Label Encoding for Gender & Purchased
df2 = df[['Gender','Purchased']]  # safer than iloc
print(df2.head(10))

le = LabelEncoder()
df2['Gender'] = le.fit_transform(df2['Gender'])
df2['Purchased'] = le.fit_transform(df2['Purchased'])
print(df2.head(10))





df2['Education_new']=le.fit_transform(df['Education'])
df2['Review_new']=le.fit_transform(df['Review'])
print(df2.head())


#apply lable encoder on nominal categorical variable

df3=df[['Gender','Purchased']]
df3['Gender_new']=le.fit_transform(df['Gender'])
df3['Purchased_new']=le.fit_transform(df['Purchased'])
print(df3.head(10))


df[['Gender','Purchased']]=df3[['Gender_new','Purchased_new']].copy()
print(df.head(2))
"""df.update(df1)
print(df.head(2))"""




