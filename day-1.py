# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:26:16 2023

@author: Pooja
"""
from platform import python_version
print(python_version())


python_version()
#Numpy version
import numpy as np
print(np.__version__)

from numpy import loadtxt
path = r"C:\Users\Pooja\iris.txt"
datapath= open(path, 'r')
data_num = loadtxt(datapath, delimiter=",")

import numpy as np
import csv
path = r"C:\Users\Pooja\student-mat.csv"
with open(path,'r') as f:
   reader = csv.reader(f,delimiter = ',')
   headers = next(reader)
   data_csv = list(reader)
   data_csv_num=np.asarray(data_csv)
   
from pandas import read_csv
path = r"C:\Users\Pooja\iris.csv"
data = read_csv(path)