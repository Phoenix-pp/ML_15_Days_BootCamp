# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:44:45 2023

@author: Pooja
Feature scaling and Regression
with explanation on ML types
"""

from sklearn import datasets, linear_model
#
# Load the Sklearn diabetes data set
#
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
data=datasets.load_diabetes()

import pandas as pd

df=pd.DataFrame(data.data, columns=data.feature_names)
df['target']=data.target
#
# Create scaled data set
#fetching bmi
raw = diabetes_X[:, None, 2]


#Min-Max scaling
"""
The goal of min-max scaling is to ensure 
that all features are 
on a similar scale ([-1,1] or [0, 1]
"""
#x_scaled = (x1 -x1_min)/(x1_max – x1_min)



max_raw = max(raw)
min_raw = min(raw)
scaled = (2*raw - max_raw - min_raw)/(max_raw - min_raw)

#StandardScaler -Z-Score normalization

#x_scaled = (x1 – x1_mean)/x1_stddev
from sklearn.preprocessing import StandardScaler
 
sc = StandardScaler()
scaled_raw=sc.fit_transform(raw)

from sklearn.preprocessing import MinMaxScaler
mm= MinMaxScaler()
scaled_raw_mm=mm.fit_transform(raw)

import timeit

def train_raw_data():
    linear_model.LinearRegression().fit(raw, diabetes_y)
 
def train_scaled_data():
    linear_model.LinearRegression().fit(scaled_raw_mm, diabetes_y)
 
#
# Use the timeit method to measure the
# execution of training method
#
raw_time = timeit.timeit(train_raw_data, number=1000)
scaled_time = timeit.timeit(train_scaled_data, number=1000)
#
# Print the time taken to
# train the model with raw data and scaled data
#
raw_time, scaled_time



"""
Supervised learning
Unsupervised Learning
Reinforcment Learning

target along with attribute
Baisc steps in ML:-
data collection
data pre-processing -
datatypes, missing, outlier, univar, mu, coore, scaling
split data :- training set, testing set, validation set
model will be trained
test the model, evaluate-feedback to feature engineering/training
dump model, deploy
------
Supervised Learning
------
Feature/attriutes, Label/Target
Classification:- KNN, DT, RFT, NB,Logistic Reg
Regression:- LR, SVM
?Type of Labels/Target 
1. Categories, Discrete - Classification-Classify
2. Continuous , Quantitative - regression -Predict
------
Un-Supervised Learning
------
Feature/attriutes -NO Taget/Label
? by understanding the relation 
between the observations/data point
clusters:- K-means
"""
import numpy as np
import statsmodels.api as sm
import pandas as pd

df=pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv', index_col=0)
df.head()
df.columns
y = df.Employed
X = df.GNP
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
est.summary()
#y=51.8436+0.0348*x
#gnp=234.289
est.fittedvalues 
res = est.resid
#create Q-Q plot
fig = sm.qqplot(res, fit=True, line='r')



