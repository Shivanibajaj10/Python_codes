#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:18:42 2023

@author: shivanibajaj
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_grubhub_edited.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Get an overview of the data
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()

# Verify that missing values have been removed
print(df.isnull().sum())

df.to_csv('cleaned_grubhub.csv')



plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Print all column names of the DataFrame
print(df.columns)

# Define your features and target variable
X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['delivery_time']

# Binary target variable for Logistic Regression
# Assuming we want to predict whether the delivery_fee is above the average fee
df['high_fee'] = (df['delivery_fee'] > df['delivery_fee'].mean()).astype(int)
y_log = df['high_fee']

# Split the data into training and testing sets for both models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prediction on test data
y_pred = lin_reg.predict(X_test)

# Evaluation
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

''' Mean Absolute Error: 9.970619481425661
Mean Squared Error: 169.68874816758134
Root Mean Squared Error: 13.026463379121033'''

log_reg = LogisticRegression()
log_reg.fit(X_train_log, y_train_log)

# Prediction on test data
y_pred_log = log_reg.predict(X_test_log)

# Evaluation
print('Accuracy:', metrics.accuracy_score(y_test_log, y_pred_log))
'''Accuracy: 0.7391857506361323'''


######################################################################################

'''Below codes are according to the project proposal "Methods" and "Objectives'''
'''1. Descriptive Statistics and Data Visualization'''

# Descriptive statistics
print(df.describe())
'''       searched_zipcode  searched_lat  ...  review_rating     high_fee
count       7857.000000   7857.000000  ...    7857.000000  7857.000000
mean       36262.899325     39.964937  ...       3.476592     0.573501
std        34128.983317      2.954739  ...       1.406192     0.494600
min         2118.000000     33.683250  ...       0.000000     0.000000
25%        10003.000000     40.631075  ...       3.470000     0.000000
50%        11219.000000     40.762983  ...       3.860000     1.000000
75%        60640.000000     41.947206  ...       4.310000     1.000000
max        92804.000000     42.457335  ...       5.000000     1.000000

[8 rows x 13 columns]'''

# Correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

'''2. Forecasting and Regression'''

# Import necessary libraries
from statsmodels.tsa.arima_model import ARIMA

# Assuming 'date' column exists and is datetime type and you want to forecast 'delivery_time'
# Convert 'RunDate' to datetime if it's not already

# Import necessary libraries
from statsmodels.tsa.arima.model import ARIMA

# Assuming 'delivery_time' column exists and is of type datetime
model = ARIMA(df['delivery_time'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)  # forecast the next 10 data points
print(forecast)

'''
7857    48.137388
7858    50.059993
7859    47.868417
7860    47.117023
7861    47.656311
7862    48.636345
7863    48.239733
7864    48.281552
7865    48.012481
7866    48.012918
Name: predicted_mean, dtype: float64

this output suggests that, for the next 10 time periods 
after the end of your training data, the delivery time would be approximately
(48.137388, 50.059993, etc.) 
(7857 to 7866) represent the forecasted 'time' or order of the data points. 
'''
'''3. Data Mining'''
# Import necessary libraries
from sklearn.cluster import KMeans

# Assuming you want to perform clustering on 'distance' and 'delivery_time'
X_cluster = df[['distance', 'delivery_time']]

# Create a KMeans object
kmeans = KMeans(n_clusters=3)  # 3 clusters for example
kmeans.fit(X_cluster)
df['Cluster'] = kmeans.labels_

# Check the result
print(df.head())

'''
                     searched_zipcode  searched_lat  ...  high_fee Cluster
RunDate                                              ...                  
2022-04-25 07:01:11             11216     40.678832  ...         0       1
2022-04-25 07:01:11             11216     40.678832  ...         0       2
2022-04-25 07:01:11             11216     40.678832  ...         0       1
2022-04-25 07:01:11             11216     40.678832  ...         0       1
2022-04-25 07:01:11             11216     40.678832  ...         0       1

[5 rows x 28 columns]
'''
'''4. Optimization'''
# Assuming 'distance' and 'delivery_time' exist
optimized_deliveries = df.sort_values(by=['distance', 'delivery_time'])

# Check the result
print(optimized_deliveries.head())
'''
                     searched_zipcode  searched_lat  ...  high_fee Cluster
RunDate                                              ...                  
2022-04-25 07:01:11             10003     40.732473  ...         0       2
2022-04-25 07:01:11              7302     40.721832  ...         0       2
2022-04-25 07:01:11             90505     33.810177  ...         0       1
2022-04-25 07:01:11              2139     42.365300  ...         0       1
2022-04-25 07:01:11             10023     40.777030  ...         1       0

[5 rows x 28 columns]
'''

'''5. Decision Tree'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

# Define your features and target variable
X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['delivery_time']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=3)

# Fit the model
dt.fit(X_train, y_train)

# Predict on the test data
y_pred = dt.predict(X_test)

# Calculate RMSE of the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
'''Root Mean Squared Error: 14.84550839200913'''

# Plot the Decision Tree

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=300)
tree.plot_tree(dt, 
               feature_names=X.columns, 
               class_names=['delivery_time'],
               filled=True)
plt.show()

######################################################################################
##This is for high/low fee

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize logistic regression model
log_reg = LogisticRegression()

# Fit the model on training data
log_reg.fit(X_train_log, y_train_log)

# Prediction on test data
y_pred_log = log_reg.predict(X_test_log)

# Evaluation
print("Accuracy:", accuracy_score(y_test_log, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test_log, y_pred_log))
print("Classification Report:\n", classification_report(y_test_log, y_pred_log))

######################################################################################
#This is for on time delivery

# Create 'ontime_delivery' column
# Convert 'delivery_time_raw' to datetime format
# Assume 'delivery_time_raw' represents minutes, convert to seconds
# Function to handle time ranges and convert to seconds
def process_time_range(time_range):
    time_range = time_range.split('-')
    avg_time = sum([float(t) for t in time_range]) / len(time_range)
    return avg_time * 60  # convert to seconds

# Apply function to 'delivery_time_raw' column
df['delivery_time_raw_seconds'] = df['delivery_time_raw'].apply(process_time_range)

# Create 'ontime_delivery' column
df['ontime_delivery'] = np.where(df['delivery_time'] > df['delivery_time_raw_seconds'], 0, 1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define your features and target variable
X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['ontime_delivery']  # our new target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier()

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

'''print("Classification Report:")
Classification Report:

print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           1       1.00      1.00      1.00      1572

    accuracy                           1.00      1572
   macro avg       1.00      1.00      1.00      1572
weighted avg       1.00      1.00      1.00      1572
'''

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
'''Confusion Matrix:
[[1572]]'''
# Accuracy
print("Accuracy:")
print(accuracy_score(y_test, y_pred))
'''Accuracy:
1.0'''

####cross validation in order to solve overfitting

from sklearn.model_selection import cross_val_score

# Define your features and target variable
X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['ontime_delivery']  # our new target variable

# Initialize the Random Forest Classifier
rf = RandomForestClassifier()

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=10)  # cv specifies the number of folds

# The cross_val_score function returns an array of accuracy scores for each fold
print("Cross-validation scores: ", scores)

# To get the average of these scores, we can use numpy's mean function
print("Average cross-validation score: ", np.mean(scores))

######################################################################################

'''Addressing the overfitting'''

#Shuffled data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['ontime_delivery']  # our new target variable

# Initialize the Random Forest Classifier
rf = RandomForestClassifier()

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=10)  # cv specifies the number of folds

# The cross_val_score function returns an array of accuracy scores for each fold
print("Cross-validation scores: ", scores)

# To get the average of these scores, we can use numpy's mean function
print("Average cross-validation score: ", np.mean(scores))

######################################################################################

##Regularization: 
    
X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['ontime_delivery']  # our new target variable

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(max_depth=5, min_samples_split=20)
rf.fit(X_train, y_train)

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=10)  # cv specifies the number of folds

# The cross_val_score function returns an array of accuracy scores for each fold
print("Cross-validation scores: ", scores)

# To get the average of these scores, we can use numpy's mean function
print("Average cross-validation score: ", np.mean(scores))

######################################################################################
## Simplify the data

X = df[['distance', 'review_count', 'review_rating', 'service_fee']]
y = df['ontime_delivery']  # our new target variable

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=10)
rf.fit(X_train, y_train)


# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=10)  # cv specifies the number of folds

# The cross_val_score function returns an array of accuracy scores for each fold
print("Cross-validation scores: ", scores)

# To get the average of these scores, we can use numpy's mean function
print("Average cross-validation score: ", np.mean(scores))

######################################################################################
#Making a decision tree for on time deliveries model 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# Drop all non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

X = df[numeric_cols].drop(columns=['ontime_delivery'])
y = df['ontime_delivery']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Fit the model
dt.fit(X_train, y_train)

# Predict on test data
y_pred = dt.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['late', 'ontime'], max_depth=3)
plt.show()







