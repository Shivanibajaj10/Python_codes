#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:52:52 2023

@author: shivanibajaj
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("BostonHousing (1).csv")
df=pd.read_csv('/Users/ShivaniBajaj/Downloads/BostonHousing.csv') #wokring for some reason
#multiple regression
x=df[["crim","chas","rm"]]
y=df["medv"]
from sklearn.linear_model import LinearRegression 
lm=LinearRegression()
lm.fit(x, y) 
print('The intercept value is:', lm.intercept_)
print('The coefficient values are:', lm.coef_)
#Using the estimated regression model above, what median house price is predicted for a tract in
#the Boston area that does not bound the Charles River, has a crime rate of 0.1, and where the
#avg number of rooms per house is 6?
pred = pd.DataFrame({'crim': [0.1], 'chas': [0], 'rm': [6]})
predicted_house_price = lm.predict(pred)
print('The predicted median house price is:', predicted_house_price[0]) 

#Compute the correlation table for the numerical predictors and search for highly correlated
#pairs. Based off the correlation table, comment on which predictors could be removed? Discuss
#the relationships among INDUS, NOX, and TAX
cormatrix = df.corr()
print(cormatrix)
cormatrix.to_csv('correlation_table.csv', index=True, header=True)

#Use three subset selection algorithms: backward, forward, and exhaustive) on all the predictors.
#Compute the testing RMSE for each of the three selected models. (Use random state=1, test
#size=0.3).
df=pd.get_dummies(df,drop_first=True)
x=df.drop(columns = "medv")
y=df["medv"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
lm=LinearRegression()
lm.fit(x_train,y_train)

y_pred = lm.predict(x_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print ("rmse is",rmse)

# Backward regression
linearr = LinearRegression()##model to be used for feature selection

sfs = SFS(linearr, 
          k_features=(1,12), 
          forward=False, 
          scoring='neg_root_mean_squared_error',
          cv=10)

sfs.fit(x_train, y_train)##training here means finding important features based on RMSE

###which features were selected
sfs.k_feature_names_

##transformed data will have selected features only
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fitting the model with the new feature subset
# and make a prediction on the test data
linearr.fit(X_train_sfs, y_train)
y_pred = linearr.predict(X_test_sfs)

'''rmse with backward method'''

rmse = mean_squared_error(y_test, y_pred, squared=False)
print ("rmse with backward method is",rmse) 
##4.55271

# Forward regression
lregression = LinearRegression()##model to be used for feature selection

sfs = SFS(lregression, 
          k_features=(1,12), 
          forward=True, 
          scoring='neg_root_mean_squared_error',
          cv=10)

sfs.fit(x_train, y_train)##training here means finding important features based on RMSE

###which features were selected
sfs.k_feature_names_

##transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fitting the model with the new feature subset
# and making the prediction on the test data
lregression.fit(X_train_sfs, y_train)
y_pred = lregression.predict(X_test_sfs)

'''rmse with forward method'''

rmse = mean_squared_error(y_test, y_pred, squared=False)
print ("rmse with forward method is",rmse) 
##4.55271

##Exhaustive searching

lr = LinearRegression()

efs = EFS(lr, 
          min_features=1,
          max_features=len(x.columns),
          scoring='neg_root_mean_squared_error',
          cv=10)

efs.fit(x_train, y_train)

##fetures selected
efs.best_feature_names_


X_train_efs = efs.transform(x_train)
X_test_efs = efs.transform(x_test)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
lr.fit(X_train_efs, y_train)
y_pred = lr.predict(X_test_efs)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print ("rmse with exhaustive method is",rmse) 
##4.55271