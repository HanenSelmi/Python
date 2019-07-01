# Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
#shops file
theshops = pd.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\theshops.csv", sep='|', encoding='latin-1' )
print (theshops.shape)
theshops.head()
data_set = theshops.values
print(data_set)
print(theshops.describe())
theshops.isnull().any()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])


print("***** Test_Set *****")
print(theshops.describe())



print('----- Data set partitioning----')
train_set = data_set[:1, : ]
print('number of instances in the training set:', train_set.shape[0])
print('number of attributes in the training set:', train_set.shape[1])
print('--------------------')



theshops.isnull().any()
print("---- the missing variables in the data set----")
print (theshops.isnull().head())
print("---- the sum of the missing variables ----")
print (theshops.isnull().sum())
print("---- the total number of the missing variables ----")
print (theshops.isnull().sum().sum())


print('------ Replacing Missing Values --------')

median= theshops.median()
print(theshops.fillna(median, inplace=True))

print()
print("----------- New data set visualization -----------")
print(theshops)
print()
print('--------------------')
print (theshops.isnull())
print()
print ('Number of missing values: ',theshops.isnull().sum().sum())
print('--------------------')

print("----- data types ------")
theshops.info()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])

