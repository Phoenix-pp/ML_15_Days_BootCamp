# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:41:30 2023

@author: Pooja
"""
import numpy as np
import pandas as pd
data=pd.read_csv('titanic.csv')
data.info()
data.describe()
att=data.columns
#attributes=list(att)
data.head(20)
data.tail()
data.survived.unique()
id_nan=data.survived[data['survived']=='NaN'].index.values
i=data.index
list(np.where(data['survived']=='NaN'))
data.dropna(inplace=True)
survived=np.array(data.survived)
np.isnan(data.age).sum()





data_1=data[['pclass','survived','age','fare','name','sex']]
data_1.shape
data_1.head()
np.isnan(data_1).sum()
data_1.dtypes
data_1.pclass.fillna(1)
data_1.age.fillna(data_1.age.mean())
data_1.survived.fillna(data_1.survived.mode())
np.isnan(data_1.survived)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_1['Sex'] = le.fit_transform(data_1['sex'])
#newdf=df
data=data.drop('home.dest', axis=1)

data=data.drop(index=1309)


data.drop("ticket",axis=1,inplace=True)
data.drop("parch",axis=1,inplace=True)
data.isnull().sum()
data.age.fillna(data.age.mean(), inplace=True)

data.Fare=data.fare.bfill()
updated_df = data[['age','pclass','survived','fare']]
updated_df['Ageismissing'] = updated_df['age']
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(missing_values='NaN',strategy = 'median')
my_imputer.fit(updated_df)
data_new = my_imputer.transform(updated_df)
updated_df.info()


import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))

import pandas as pd
data.age.hist()
data.pclass.hist()
import matplotlib.pyplot as plt
plt.plot(data.age[900:])
plt.hist(data.age)
data.age[1308]=200
plt.boxplot(data.age)
plt.show()
plt.close()

fig, ax = plt.subplots(1,2)
ax[1,1].plot(data.age)
ax[1,2].plot(data.age[900:])

import matplotlib.pyplot as plt
import pandas as pd
# Load the Titanic dataset
data = pd.read_csv("titanic.csv")

# Plot a box plot of the age of passengers, grouped by their survival status
x=data[data['survived'] == 0]['age'].dropna()
y=data[data['survived'] == 1]['age'].dropna()
plt.boxplot([x, y], labels=['Did not survive', 'Survived'],
            patch_artist=True, sym='ro', notch=True, whis=True)

# Add labels and title
plt.xlabel('Survival Status')
plt.ylabel('Age')
plt.title('Box Plot of Age by Survival Status')

# Show the plot
plt.show()

plt.savefig("plot.png", dpi=300, bbox_inches='tight', pad_inches=0.5, transparent=True)

plt.savefig("plot.png")


plt.savefig("filename.extension")

plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png")
"""
This will save the plot as a PNG file with the name "example.png".

    dpi:

"""

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", dpi=300)

"""This will save the plot as a PNG file with the name "example.png" and a resolution of 300 DPI.

    facecolor and edgecolor:

python"""

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", facecolor='red', edgecolor='blue')
"""
This will save the plot as a PNG file with the name "example.png" and a red background color and a blue border color.

    orientation:
"""

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", orientation='landscape')

This will save the plot as a PNG file with the name "example.png" in landscape orientation.

    papertype:

python

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", papertype='legal')

This will save the plot as a PNG file with the name "example.png" on legal size paper.

    format:

python

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.pdf", format='pdf')

This will save the plot as a PDF file with the name "example.pdf".

    transparent:

python

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", transparent=True)

This will save the plot as a PNG file with the name "example.png" with a transparent background.

    bbox_inches:

python

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", bbox_inches='tight')

This will save the plot as a PNG file with the name "example.png" with a tight bounding box around the plot.

    pad_inches:

python

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.savefig("example.png", pad_inches=0.5)

This will save the plot as a PNG file with the name "example.png" with a 0.5-inch padding around the plot.

plt.savefig('plot.pdf')






import matplotlib.pyplot as plt 
x = [1, 2, 3, 4] 
y = [2, 4, 6, 8] 
fig, ax = plt.subplots()
ax.plot(x, y, color ='blue', linestyle='dashed', marker='o', linewidth=2.0)
ax.set(xlim=(0, 5), xticks=np.arange(1, 5),
       ylim=(0, 10), yticks=np.arange(1, 10), 
       title="Plot of x versus y", 
       xlabel="Values of x", ylabel="Values of y", label='y')
ax.annotate('3,6 point', xy=(3.1, 5.9), xytext=(3.5, 5.5),
            arrowprops=dict(facecolor='red', shrink=0.05))
ax.grid(True, linestyle='--')
ax.tick_params(labelcolor='r', labelsize='medium', width=3)

plt.show() 






