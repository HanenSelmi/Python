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
print("-------------DATA SET 2  OF SHOPS----------")
l2 = pd.read_csv("C:\Users\TOSHIBA\Desktop\ml-latest-small\store.csv",sep='\t')

data_set = l2.values
print(data_set)
print(l2.describe())

