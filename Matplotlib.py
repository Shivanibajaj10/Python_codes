#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:39:38 2023

@author: shivanibajaj
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df5=pd.read_excel('PCA_Supplementary.xlsx', sheet_name='Sheet1')
sns.scatterplot(x='X1',y='X2',data=df5)
sns.scatterplot(x='std_x1',y='std_x2',data=df5)
sns.scatterplot(x='PC1',y='PC2',data=df5)
p=df5.corr()

df6=pd.read_excel('foodprices.xlsx')
df7=df6.drop(columns='City')
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(df7)
scaled_df=scaler.transform(df7)
from sklearn.decomposition import PCA#import
PCA=PCA(n_components=2)#initialize
PCA.fit(scaled_df)#train
PCA_df5=pca.transform(scaled_df5)#transform
PCA.explained_variance_ratio_

PCA_df5=pd.DataFrame(pca_df5,columns=['PC1','PC2'])
totalset=pd.concat([pca_df5,df6],axis=1)

#correlation b/w the old and new varibles

loadings=totalset.corr()
plt.figure(figsize=(10,10))
sns.scatterplot(x='PC1',y='PC2',data=totalset)
for i in range(len(totalset)):
    plt.text(totalset.loc[i,'PC1'],totalset.loc[i,'PC2'],totalset.loc[i,'City'])
plt.xlabel('CPI for non-fruits (pc1)')
plt.ylabel('negative CPI for fruits (pc2)')

# 2 interesting rules - 
#  alot of people who bought french fries, bought avacado as well
#  akot of people who bought fgreen tea, bought brownies as well