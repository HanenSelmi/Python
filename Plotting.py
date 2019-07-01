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
df = pandas.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\users.csv",sep=";", names=['user_id','gender','age','zipcode','occupation'])
data_set = df.values
print( df  )

#Check the data types of different features that we have
df.info()
#convert character column to numeric
df['user_id'] = pandas.to_numeric(df['user_id'],errors='coerce')
df['gender'] = pandas.to_numeric(df['gender'],errors='coerce')
df['age'] = pandas.to_numeric(df['age'],errors='coerce')
df['zipcode'] = pandas.to_numeric(df['zipcode'],errors='coerce')
df['occupation'] = pandas.to_numeric(df['occupation'],errors='coerce')
#recheck
df.info()

print("------------------------------------------------------------------")


# Getting the values and plotting it
f1 = df['age'].values
f2 = df['gender'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='red', s=5)
plt.show()
