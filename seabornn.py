import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\sampledata\\EDA_data.csv")
print(sns.boxplot(df['EstimatedSalary'],orient='h'))
plt.show()