# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:23:32 2023

@author: Pooja
"""
import numpy as np
c = np.array([[(1.5,2,3), (4,5,6)],[(3,2,1), (4,5,6)]], dtype = float)
c_l=c.tolist()
c_f=c.flatten()
c_r=c.repeat(1)
x = np.array([[[0], [1], [2]]])
x.shape
np.squeeze(x).shape
c.sort()
c.argsort()
c.max()
c.argmax()
c[1,1]=np.nan

np.isnan(c).sum()
np.nan_to_num(c)
np.nanmax(c)
c[np.isnan(c)]=np.nanmax(c)

from numpy import loadtxt
path = r"C:\Users\Pooja\iris.txt"
datapath= open(path, 'r')
data_num = loadtxt(datapath, delimiter=",")


data1=np.genfromtxt(path, delimiter=',',skip_header=False)

np.save('arr', c)
np.savez('arr1',c,data_num)
data_load=np.load('arr1.npz')

hist,edges=np.histogram(data_num[1])

hist,edges=np.histogram2d(data_num[1],data_num[2])

import matplotlib.pyplot as plt
plt.hist(data_num[1])
np.unique(c)

import pandas as pd
data=pd.read_csv('titanic.csv')
data.info()
data.describe()
attributes=list(data.columns)
data.head(20)
data.tail()
data.survived.unique()
survived=np.array(data.survived)
np.isnan(data.age).sum()























