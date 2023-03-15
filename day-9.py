# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:38:56 2023

@author: Pooja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('w-h.csv')
df.describe()
#Visualizing the data attribute height
plt.hist(df.Height)
#Visualizing using boxplot for outlier detection
plt.boxplot(df.Height)
#setting up the limits to remove the outlier using quantile
max_height=df.Height.quantile(0.95)
min_height=df.Height.quantile(0.05)
#Displaying rows in dataset beyond the set limits
print(df.Height[df.Height>max_height].count())
print(df.Height[df.Height<min_height].count())
#removing the outliers in the dataset
df_Height_without_outlier=df[(df.Height<max_height) & (df.Height>min_height)]
#Displaying the datset attribute with and without ourlier using boxplot
plt.subplot(1,2,1)
plt.boxplot(df.Height)
plt.title("With outliers")
plt.subplot(1,2,2)
plt.boxplot(df_Height_without_outlier.Height)
plt.title("Without outliers")
plt.show()

from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_excel('Iris1.xls')
df.Species.head()

mean_map = df.groupby('species')['Species'].mean().to_dict()
df['color_target_mean'] = df['color'].map(mean_map)

import category_encoders as ce
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create a DataFrame from the dataset
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Select a categorical variable to encode
cat_var = 'worst concave points'

# Create a target encoder and fit on the categorical variable
encoder = ce.TargetEncoder(cols=cat_var)
encoder.fit(df[cat_var], df['target'])

# Transform the categorical variable
df_encoded = encoder.transform(df[cat_var])

# Add the transformed variable to the original DataFrame
df[f'{cat_var}_target_encoded'] = df_encoded

# Show the first few rows of the encoded DataFrame
print(df.head())


import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['Species'])
df_encoded = encoder.fit_transform(df.Species)
df_encoded.head()

from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_excel('Iris1.xls')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['Species'])
print("Original categories \n", df.Species.unique())
print("categories encoded as numerics \n",df.species_encoded.unique())

from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_excel('Iris1.xls')
count_map = df['Species'].value_counts().to_dict()
df['Species_count'] = df['Species'].map(count_map)
df[['Species','Species_count']]






from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_excel('Iris1.xls')
one_hot = pd.get_dummies(df['Species'])
df = pd.concat([df, one_hot], axis=1)
one_hot.columns


df_mean=d.loc['mean']
plt.hist(df.Weight,bins=20, edgecolor='black')

from scipy.stats import norm
plt.hist(df.Height,bins=20, edgecolor='black')
r=np.arange(df.Height.min(),df.Height.max())
plt.plot(r, norm.pdf(r, df.Height.mean(),df.Height.std()), color='r')
plt.hist(df.Height,bins=20, edgecolor='black')
plt.show()

df.age.iloc[23]=120
ul=df.Height.mean()+3*df.Height.std()
ll=df.Height.mean()-3*df.Height.std()
df['zscore_height']=(df.Height-df.Height.mean())/df.Height.std()
df[df['zscore_height']>3]
df[df['zscore_height']<-3]
df_with_o_z=df[(df['zscore_height']>-3) & (df['zscore_height']<3)]


dataframe.describe()
from pandas.plotting import scatter_matrix

scatter_matrix(dataframe)

