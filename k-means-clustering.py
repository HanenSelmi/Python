# Dependencies
import pandas as pandas
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



print("Handling the data set")
df = pandas.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\users.csv",sep=";", names=['user_id','gender','age','zipcode'])
data_set = df.values
print('------ Replacing Missing Values --------')
median= df.median()
print(df.fillna(median, inplace=True))

#Check the data types of different features that we have
df.info()
#convert character column to numeric
df['user_id'] = pandas.to_numeric(df['user_id'],errors='coerce')
df['gender'] = pandas.to_numeric(df['gender'],errors='coerce')
df['age'] = pandas.to_numeric(df['age'],errors='coerce')
df['zipcode'] = pandas.to_numeric(df['zipcode'],errors='coerce')

df['gender'] = df['gender'].astype(float)
df['age'] = df['age'].astype(float)
df["age"].fillna("18", inplace = True)
#recheck
df.info()

print("------------------------------------------------------------------")



X = np.array(df.drop(['age'], 1).astype(float))
y = np.array(df['gender'])
kmeans = KMeans(n_clusters=8)

kmeans.fit(X)
print(kmeans.fit(X))
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

Y = df[['age']]
X = df[['gender']]
# Principal Component Analysis
pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)
# KMeans

KMeans = KMeans(n_clusters=3)

KMeans_output = kmeans.fit(Y)

KMeans_output

plt.figure('3 Cluster K-Means')

plt.scatter(pca_c[:, 0], pca_d[:, 0], c=KMeans_output.labels_)

plt.xlabel('gender')

plt.ylabel('age')

plt.title('3 Cluster K-Means')
plt.show()


