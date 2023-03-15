# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:22:55 2023

@author: Pooja
"""



"""
Logistic regression is a popular machine learning algorithm used for binary classification tasks. 
The goal of binary classification is to predict a binary output (e.g., yes/no, true/false) 
based on one or more input variables. 
For example, logistic regression could be used to 
predict whether a customer will churn (cancel their subscription) or not, 
based on their demographics and purchase history."
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
titanic_df = pd.read_csv('titanic.csv')
titanic_df=titanic_df.drop(1309)
# Remove irrelevant features
titanic_df.columns
titanic_df = titanic_df.drop(['pclass', 'name', 'ticket', 'cabin', 'embarked'], axis=1)

# Replace missing values with the mean value
titanic_df['age'].fillna(titanic_df['age'].mean(), inplace=True)

# Convert categorical features to one-hot encoding
titanic_df = pd.get_dummies(titanic_df, columns=['sex'], drop_first=True)

# Split the dataset into training and testing sets
X = titanic_df.drop('survived', axis=1)
X=X.drop('home.dest', axis=1)
X=X.drop('boat', axis=1)
X=X.drop('body', axis=1)
X.fillna(X.mean())
y = titanic_df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the logistic regression model
model = LogisticRegression(random_state=(16)).fit(X_train, y_train)
model.fit(X_train, y_train)
model.score(X_train, y_train)

# Evaluate the performance on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
#The .2f format specifier is used to format the accuracy variable as a floating-point number with two decimal places. 
#This means that the printed value of accuracy will have two digits after the decimal point.
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1))

# Make predictions on new data
new_data = pd.DataFrame({'Pclass': [3], 'Age': [25], 'SibSp': [1], 'Parch': [0], 'Fare': [7.5], 'Sex_male': [1]})
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print(prediction)
