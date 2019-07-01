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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


ratingss = pandas.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\ratingss.csv" ,sep=';', encoding='latin-1', names=['userId','shop_Id','rating'] )
#shops file
theshops = pandas.read_csv("C:\\Users\TOSHIBA\Desktop\\IT\\theshops.csv", sep=';', encoding='latin-1', names=['shop_Id','name_id','category'] )
print("--------------------------------")
theshops['shop_Id'] = pandas.to_numeric(theshops['shop_Id'],errors='coerce')
ratingss['shop_Id'] = pandas.to_numeric(ratingss['shop_Id'],errors='coerce')
ratingss['userId'] = pandas.to_numeric(ratingss['userId'],errors='coerce')
ratingss['rating'] = pandas.to_numeric(ratingss['rating'],errors='coerce')
#merge
shop_data = pandas.merge(ratingss, theshops, on='shop_Id')
shop_data.head()
print(shop_data.describe())
print("***********")

