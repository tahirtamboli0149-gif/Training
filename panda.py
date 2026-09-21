'''Data manipulation with pandas.'''
import pandas as pd
print(pd.__version__)

#converting list to series
l1=[10,20,'pune','iccd']
s1=pd.Series(l1)
print(s1)
print(type(s1))

#changing index
s1=pd.Series(l1, index=['a','b','c','d'])
print(s1)
print(type(s1))


#dictionary to series
d={'a':1, 'b':2, 'c':3}
s2=pd.Series(d)
print(s2)

#slicing dictionary
s2[2:4]
print(s2[2:4])

#minmax
print(s2.min())
print(s2.max())

'''DataFrame: Two dimentional label array'''


l1=[[1,2,3],[4,5,6],[7,8,9]]
d1=pd.DataFrame(l1)
print(d1)

#from list of dictionary
l2=[{'a':1,'b':2},{'a':3,'b':4}]
d2=pd.DataFrame(l2)
print("\n",d2)


#importing file from sysytem
df=pd.read_csv("C:\\sampledata\\EDA_data.csv")
print(df.head())
print(df)

print(df.shape)

print(df.size)

print(df.info())

print("\n",df.describe())


 