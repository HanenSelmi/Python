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

#let's take a look at the average rating of each shop
#we can group the dataset by the name of the shop and then calculate the mean of the rating for each shop.
# We will then display the first five shops along with their average rating using the head() method
print(shop_data.groupby('name_id')['rating'].mean().head())

#sort the ratings in the descending order of their average ratings:
print(shop_data.groupby('name_id')['rating'].mean().sort_values(ascending=False).head())
print("------------")
#plot the total number of ratings for a shop:
print(shop_data.groupby('name_id')['rating'].count().sort_values(ascending=False).head())

ratings_mean_count = pandas.DataFrame(shop_data.groupby('name_id')['rating'].mean())
print(ratings_mean_count)


#create ratings_mean_count dataframe and first add the average rating of each shop to this dataframe
ratings_mean_count['rating_counts'] = pandas.DataFrame(shop_data.groupby('name_id')['rating'].count())


#Now let's take a look at our newly created dataframe.
print(ratings_mean_count.head() )

#plot a histogram for the number of ratings represented by the "rating_counts"
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=10)

#histogram of histogram for average ratings
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating'].hist(bins=10)


#We will plot average ratings against the number of ratings
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)
plt.show()

print("****************************************")
# the matrix of shops titles and corresponding user
user_shop_rating = shop_data.pivot_table(index='userId', columns='name_id', values='rating')
user_shop_rating.head()
print(user_shop_rating)

#find the user ratings for Taccos Chaneb
print ("The user ratings for Taccos Chaneb")
forrest_ratings = user_shop_rating['Tacos chaneb']
forrest_ratings.head()
print(forrest_ratings)

#similar shops/correaltion
shopss_like_tacos = user_shop_rating.corrwith(forrest_ratings)
print ("correlation")
corr_tacos = pandas.DataFrame(shopss_like_tacos, columns=['Correlation'])
corr_tacos.dropna(inplace=True)
corr_tacos.head()
print(corr_tacos)

#shops in descending order of correlation to see highly correlated movies at the top
print(corr_tacos.sort_values('Correlation', ascending=False).head(10) )

#retrieve only those correlated movies that have at least more than 10 ratings
corr_forrest_gump = corr_tacos.join(ratings_mean_count['rating_counts'])
corr_forrest_gump.head()
print(corr_forrest_gump)

#more than 30000 ratings
print(corr_forrest_gump[corr_forrest_gump ['rating_counts']>30000].sort_values('Correlation', ascending=False).head(100))
