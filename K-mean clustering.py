import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# reading dataset
df = pd.read_csv('C:\\sampledata\\student_clustering.csv')
print(df.head(10))
print("\nDescribing the dataset:")
print("\n", df.describe())
print(df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.info())

# visualizing data clusters
plt.scatter(df['cgpa'], df['iq'])
plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Scatter plot of students")
# plt.show()

"""APPLYING K-MEAN CLUSTERING"""
WCSS = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, random_state=42)
    km.fit(df[['cgpa', 'iq']])   # only use numeric features
    WCSS.append(km.inertia_)

print("WCSS values:", WCSS)

 #inertia_  measures how eans.
 #IT is calculated by measuring the distance between each data point and its ....
 # A good model os one with low inertia AND a low nimber of clusters   




# Elbow method visualization
plt.plot(range(1, 11), WCSS, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()



x = df[['cgpa', 'iq']].values
# it is a numpy array which has cgpa and iq values

km = KMeans(n_clusters=4, n_init=10, random_state=42)

y_means = km.fit_predict(x)
# fit_predict will train the model and assign cluster values
# y_means contains cluster numbers: 0, 1, 2, 3


# Printing values of cluster 3
print(x[y_means == 3, 0])
print(x[y_means == 3, 1])


# Scatter plot of all 4 clusters

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], label='Cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], label='Cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], label='Cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], label='Cluster 4')


# Plotting centroids
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=100,
            marker='*',
            label='Centroids')


plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("K-Means Clustering of Students")
plt.legend()
plt.show()