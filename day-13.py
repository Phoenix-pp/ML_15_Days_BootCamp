# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:32:40 2023

@author: Pooja
"""
import pandas as pd
url="https://github.com/sjwhitworth/golearn/blob/master/examples/datasets/tennis.csv"
data=pd.read_csv(url)


"""
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 18:24:20 2022

@author: Pooja
"""
"""

#import all the packagesimport pandas as pdimport numpy as npimport matplotlib.pyplot as pltfrom sklearn.naive_bayes import GaussianNBfrom sklearn.preprocessing import LabelEncoderfrom sklearn.preprocessing import Normalizerfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import confusion_matrix,accuracy_scoreimport sklearn.metricsimport statsmodels.api as smimport plotly.express as px #for plotting the scatter plotimport seaborn as sns #For plotting the dataset in seabornsns.set(style='whitegrid')import warningswarnings.filterwarnings('ignore')

#import all the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn.metrics
import statsmodels.api as sm
import plotly.express as px #for plotting the scatter plot
import seaborn as sns #For plotting the dataset in seaborn
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings('ignore')

#Read the excel file
data=pd.read_csv('play_tennis.csv')
print(data.head(5))
print(data.describe())
#Data manipulation and cleaning:

# checking if data imputation is required:
print(data.columns)
data.isnull().any()
# No null value is present in the dataset but if there was one we could've used:
# data = data.dropna(axis = 0, how ='any')
data=data.drop(['day'],axis=1)

#Label encoding:
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['outlook_encoded']= le.fit_transform(data['outlook'])
data['temp_encoded']= le.fit_transform(data['temp'])
data['humidity_encoded']= le.fit_transform(data['humidity'])
data['wind_encoded']= le.fit_transform(data['wind'])
data['play_encoded']= le.fit_transform(data['play'])
print(data.head(5))

#Feature selection:

# heatmap of correlation matrix with annotations in 2 different shades
cols=['outlook_encoded','temp_encoded','humidity_encoded','wind_encoded','play_encoded']
cor=data[cols].corr()
hm1 = sns.heatmap(cor, annot = True,cmap='YlGnBu')
hm1.set(xlabel='\nFeatures', ylabel='Features\t', title = "Correlation matrix of data\n")
plt.show()


#Splitting the data into testing and training data:

features=['outlook_encoded', 'temp_encoded', 'humidity_encoded','wind_encoded']
x=data[features]# since these are the features we take them as x
y=data['play_encoded']# since play is the output or label we'll take it as y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=1)
print("\nShape of x_train:\n{}".format(x_train.shape))
print("\nShape of x_test:\n{}".format(x_test.shape))
print("\nShape of y_train:\n{}".format(y_train.shape))
print("\nShape of y_test:\n{}".format(y_test.shape))

x_train.describe()

#Scaling the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)
print(x_train)
print("____________________________________________________________________________")
print("",scaled_x_train)
## Before
# View the relationships between variables; color code by play
before= sns.pairplot(data[cols], hue= 'play_encoded')
before.fig.suptitle('Pair Plot of the dataset Before scaling', y=1.08)
## After:
data2= pd.DataFrame(data= np.c_[scaled_x_train, y_train],
columns= features + ['play_encoded'])
after= sns.pairplot(data2, hue= 'play_encoded')
after.fig.suptitle('Pair Plot of the dataset After scaling', y=1.08)

#Training the model

x_train=scaled_x_train
x_test=scaled_x_test
model = GaussianNB()
model.fit(x_train, y_train)
x_test1=np.array([[0.12,1.0,0.9,0.8]])
y_prediction= model.predict(x_test1)
report=pd.DataFrame()
report['Actual values']=y_test
report['Predicted values']= y_prediction
print(report)

#Model evaluation:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
ConfusionMatrix=confusion_matrix(y_test,y_prediction)
print(ConfusionMatrix)
ax=sns.heatmap(ConfusionMatrix,annot=True,cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
#Ticket labels:
ax.xaxis.set_ticklabels(['No','Yes'])
ax.xaxis.set_ticklabels(['No','Yes'])
plt.show()

#Classification report:

from sklearn.metrics import classification_report
print(classification_report(y_test, y_prediction))





from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_jobs=2)
RF=rd.fit(x_train,y_train)
pre_rd=rd.predict(x_test)
print(accuracy_score(y_test,pre_rd))
print(confusion_matrix(y_test, pre_rd))
print(classification_report(y_test,pre_rd))

x_train=scaled_x_train
x_test=scaled_x_test
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred=dt.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))

from sklearn.svm import SVC
sc=SVC()
sc.fit(x_train,y_train)
pred=sc.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))


