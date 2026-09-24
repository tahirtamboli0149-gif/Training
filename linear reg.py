import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#importing model
from sklearn.linear_model import LinearRegression
#importing evolutin metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df=pd.read_csv('C:\\sampledata\\placement_new.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n",df.describe())


x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
print("Shape of x",x.shape)
print("Shape of y",y.shape)

#visulizing the cgpa vs package 
plt.scatter(x=df['cgpa'], y=df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('CGPA vs Package')
#plt.show()

#spliting data into train and test parts
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
print("shpe of x_train",x_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_train.shape)
print("shape of y_test",y_test.shape)

#deploying model on training data set
lr=LinearRegression()
lr.fit(x_train,y_train)

#cheking result or score on training data
print("\naccurecy of trained dataset")
print(lr.score(x_test,y_test))


"""CALCULATING THE SLOPE USING Y=mx+b"""
#slope(m) coef_
m=lr.coef_
print(m)


#intercept
b=lr.intercept_
print(b)


print("\n",x_test.head())
print("\n",y_test.head())


y=m*7.57+b
print("\n",y)


#predict all x_tests
y_pred=lr.predict(x_test)
print("\n",y_pred)



"""VISULIZING THE PREDICTION"""

plt.scatter(df["cgpa"],df['package'])
plt.scatter(x_train,lr.predict(x_train),color='red')
#plt.show()


"""linear model evolutinn matrix
    1. lost function
    2. cost function
    3. MAE
    4. MSE
    5. RMSE
    6. R2 Score"""


#MAE
print("\nMAE",mean_absolute_error(y_test,y_pred))

#MSE
print("\nMSE",mean_squared_error(y_test,y_pred))

#RMSE
print("\nRMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

#R2_Score
print("\nR2_Score",r2_score(y_test,y_pred))









