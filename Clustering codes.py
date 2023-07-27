#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:12:10 2023

@author: shivanibajaj
"""
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_excel("assignment.xlsx")
#data descriptions
df.info()
df.head()
df.describe()

df.isna().sum()

df = df.iloc[: , 1:] #starting the data from second column  same as drop
df= df.drop("University name",axis=1) #dropping column 

df.columns

#data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df)
scaled_df=scaler.transform(df)

#Use K means to find clusters in the data set, please also interpret the clusters formed.

'''part 1'''
wcv=[]
silk_score=[]

for i in range(2,15):
    km4=KMeans(n_clusters=i,random_state=0) #initialize
    km4.fit(scaled_df) #train: identify clusters
    wcv.append(km4.inertia_)
    score = silhouette_score(scaled_df, km4.labels_)
    # Appending the score to the silk_score list
    silk_score.append(score)
    
##plotting the wcv 
plt.plot(range(2,15),wcv)
plt.xlabel('No of clusters')
plt.ylabel('within cluster variation')

#silhoutte-score
plt.plot(range(2,15), silk_score)
plt.xlabel('No of clusters')
plt.ylabel('Silk_score')

#I chose to go with 6 clusters
km3=KMeans(n_clusters=6,random_state=0) #initialize
km3.fit(scaled_df) #train: identify clusters

df['labels']=km3.labels_

##interpreting the results
sb = df.groupby('labels').mean()

df.loc[df["labels"] == 0].describe()
df.loc[df["labels"] == 1].describe()
df.loc[df["labels"] == 2].describe()
df.loc[df["labels"] == 3].describe()
df.loc[df["labels"] == 4].describe()
df.loc[df["labels"] == 5].describe()


'''part 2'''

# Now, dendrogram to find the optimal number of clusters
linked = linkage(scaled_df, 'ward')#gets a n-1 *4 matrix
dendrogram(linked) #uses the matrix to get to draw the dendrogram
plt.title("Dendrogram")
plt.xlabel('Customer')
plt.ylabel('euclidean')
plt.show()

#going with 4 here
hc=AgglomerativeClustering(n_clusters=4,linkage='ward')
hc.fit(scaled_df)


#adding labels to df
df['labels']=hc.labels_

#interpreting the results of the clusters
sn = df.groupby('labels').mean()

df.loc[df["labels"] == 0].describe()
df.loc[df["labels"] == 1].describe()
df.loc[df["labels"] == 2].describe()
df.loc[df["labels"] == 3].describe()