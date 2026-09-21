import seaborn as sns
import pandas as pd

df=pd.read_csv("C:\\sampledata\\EDA_data.csv")
print(sns.boxplot(df['EstimatedSalary'],orient='h'))
    