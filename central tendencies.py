#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:26:51 2023

@author: shivanibajaj
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

df = pd.read_excel('/Users/shivanibajaj/Downloads/National Football League.xlsx')


print(df.head())

plt.scatter(df['Yards'], df['Points'])
plt.xlabel('YardsGame')
plt.ylabel('PointsGame')
plt.title('Scatter plot of Points/Game vs Yards/Game')
plt.show()

df['intercept'] = 1
models = sm.OLS(df['Points'], df[['intercept', 'Yards']])
results = models.fit()

print(results.summary())

df1 = pd.read_excel('/Users/shivanibajaj/Downloads/Unemployment Rates.xlsx')


df1['MA_3'] = df1['Rate'].rolling(window=3).mean()
df1['MA_12'] = df1['Rate'].rolling(window=12).mean()

# Droping rows with NaN values resulted from moving average calculation
df1 = df1.dropna()

# Calculating the absolute errors for 3-month and 12-month forecasts
df1['AE_3'] = np.abs(df1['Rate'] - df1['MA_3'])
df1['AE_12'] = np.abs(df1['Rate'] - df1['MA_12'])

# Calculating Mean Absolute Deviation (MAD) for 3-month and 12-month forecasts
MAD_3 = df1['AE_3'].mean()
MAD_12 = df1['AE_12'].mean()

print(f"3-month moving average MAD: {MAD_3}")
print(f"12-month moving average MAD: {MAD_12}")







