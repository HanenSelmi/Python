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

#Reading users file:

users = pd.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\users.csv", sep='|')
print (users.shape)
users.head()
data_set = users.values
print(data_set)
print(users.describe())
users.isnull().any()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])


print("***** Test_Set *****")
print(users.describe())

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])

print('----- Data set partitioning----')
train_set = data_set[0, :]
print('number of instances in the training set:', train_set.shape[0])
print('number of attributes in the training set:', train_set.shape[0])
print('--------------------')



users.isnull().any()
print("---- the missing variables in the data set----")
print (users.isnull().head())
print("---- the sum of the missing variables ----")
print (users.isnull().sum())
print("---- the total number of the missing variables ----")
print (users.isnull().sum().sum())


print('------ Replacing Missing Values --------')

median= users.median()
print(users.fillna(median, inplace=True))

print()
print("----------- New data set visualization -----------")
print(users)
print()
print('--------------------')
print (users.isnull())
print()
print ('Number of missing values: ',users.isnull().sum().sum())
print('--------------------')

print("----- data types ------")
users.info()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])
