import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#importing model
from sklearn.tree import DecisionTreeClassifier
#imporing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#importing matrix evolution
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

#reading dataset
df = pd.read_csv('C:\\sampledata\\breast_cancer.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print(df.isnull().sum().sum())
print(df.info())   


#defining input and output
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
print("Shape of x", x.shape)
print("Shape of y", y.shape)


#splitting data into train test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=51)
print("\nSplitting data")
print("shape of x_train", x_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


#applying Decision Tree model
dc = DecisionTreeClassifier(criterion='gini', max_depth=3)
dc.fit(x_train, y_train)

#checking score
print("\nDecision Tree score of tested data")
print(dc.score(x_test, y_test))


#applying Random Forest model
rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=51)
rf.fit(x_train, y_train)

#checking score
print("\nRandom Forest score of tested data")
print(rf.score(x_test, y_test))



#evolutin matrix classification model
y_pre=dc.predict(x_test)
cm=confusion_matrix(y_test,y_pre)
print("\n",y_pre)




#visualising the prediction
sns.heatmap(cm,annot=True,fmt='g')
plt.xlabel('actual')
plt.ylabel('predicted')
#plt.show()


"""NOW FINDING:
   1.ACCURECY
   2.PRECESION
   3.RECALL
   4.F2 SCORE"""

#accurecy of model
a=accuracy_score(y_test,y_pre)
print("\n+Accurecy:",a)


#precesion of model
p=precision_score(y_test,y_pre)
print("\nPrecesion:",p)


#Recall of model
r=recall_score(y_test,y_pre)
print("\nRecall:",r)


#F1 of model
f=f1_score(y_test,y_pre)
print("\nF1 score:",f)