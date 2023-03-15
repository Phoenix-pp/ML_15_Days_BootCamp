# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:26:46 2023

@author: Pooja
"""
import sklearn
"""
step 1. import the classifier
step2. dataset - split -  training
testing
step 3 fit transform training set
step try to predict on test set, 
step accuracy, confussion matrix
error pedicted, actual prediction
"""
from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_excel('Iris1.xls')
one_hot = pd.get_dummies(df['Species'])
df = pd.concat([df, one_hot], axis=1)
one_hot.columns

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Species=le.fit_transform( df.Species)
df.columns

X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y=df.Species
# split the data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test= train_test_split(X,Y,test_size=.30)
knn=KNeighborsClassifier(n_neighbors=7, p=2)
knn.fit(x_train, y_train)
predictions=knn.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
cm = metrics.confusion_matrix(y_test, predictions, labels=knn.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
disp.plot() 



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#binary, multinominal, ordinal
# Train the logistic regression model
model = LogisticRegression(random_state=(16), multi_class='multinomial', class_weight='balanced').fit(x_train, y_train)

# Evaluate the performance on the testing set
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")

cm=confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
disp.plot() 






