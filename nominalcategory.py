import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("C:\\sampledata\\Nashik_House Price.csv")
print(df)
print(df.isnull().sum().sum())

df1=df.select_dtypes(include='str')
print(df1.head(2))

df1=df1.drop('Flat_type',axis=1)
print(df1.head(2))

for x in df1.columns:
    print(x,":",len(df1[x].unique()))

#applying pd.getdummies

dummy_df=pd.get_dummies(df1)
dummy_df=dummy_df.astype(int)
print(dummy_df.head(2))
print(dummy_df.shape)


"""ONEHOTCODER"""
oe_enc=OneHotEncoder(sparse_output=False)
print(oe_enc)
df1_arr=oe_enc.fit_transform(df1)
print(df1_arr)


"""#dropping dummy columns"""
oe_enc=OneHotEncoder(sparse_output=False,drop='first')
df1_arr=oe_enc.fit_transform(df1)
print(df1_arr)



print(dummy_df.keys())
df1_df=pd.DataFrame(df1_arr,columns=[ 'City_Pune',  'Area_Gangapur Road',
       'Area_Govind Nagar', 'Area_Wadgaon_BK', 'Area_Wagoli',
       'Parking_Yes'])
print(df1_df.head())