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
#Ratings file
print("Handling the data set")
r_cols = ['user_id', 'movie_id', 'rating']
ratingss = pd.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\ratingss.csv")

ratingss.head()
data_set = ratingss.values
print(data_set)
print( ratingss.describe())

print(ratingss.columns)


ratingss.isnull().any()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])


print("***** Test_Set *****")
print(ratingss.describe())

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])

#testing NaN values
ratingss.isnull().any()
print("---- the missing variables in the data set----")
print (ratingss.isnull().head())
print("---- the sum of the missing variables ----")
print (ratingss.isnull().sum())
print("---- the total number of the missing variables ----")
print (ratingss.isnull().sum().sum())


print('------ Replacing Missing Values --------')

median= ratingss.median()
print(ratingss.fillna(median, inplace=True))

print()
print("----------- New data set visualization -----------")
print(ratingss)
print()
print('--------------------')
print (ratingss.isnull())
print()
print ('Number of missing values: ',ratingss.isnull().sum().sum())
print('--------------------')

print("----- data types ------")
ratingss.info()

print("------- data set visualization --------")
print(data_set)
print(' number of instances:', data_set.shape[0])
print(' number of attributes:', data_set.shape[1])




print("-----------------------------------------------------------------------------------------------------------------")
#shops file
theshops = pd.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\theshops.csv", sep='|', encoding='latin-1' )
print (theshops.shape)
theshops.head()
data_set = theshops.values
print(data_set)
print(theshops.describe())

#Reading users file:

users = pd.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\users.csv", sep='|')
print (users.shape)
users.head()
data_set = users.values

