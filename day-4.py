# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:33:05 2023

@author: Pooja
"""
l=['a','b','c','d']
l1=[1,2,3,4,5]

l.extend(l1)


for i in range(len(l)):
    print(type(l[i]), end=" ")

for i in range(len(l1)):
    print(l1[i]*2)
    
l_com= [l1[x]*2 for x in range(len(l1))]  

l_even=[x for x in range(100, 200, 2)]
l_com.sort(reverse=True)
l=['a','e','y','b']
l.sort()
l.clear()
l_copy=l_com.copy()
l_com.append(23)
l_copy1=l_com
l_com[1]='hi'
l_com.pop()
l_com.pop(1)
l_com.append('hello')
l_com.remove('hello')
#---------------
#dictionary
#{key:value}
l=[1,2,3,4,5]
l1=['one','two','three','four','five']
#Tupple ()
l2=[(1,"One"),(2,"Two")]
d=dict(l2)
l3 = [(1,['a','b']),(2,['c','d'])]
d1=dict(l3)

d1.keys()
d1.values()

for i in d1.keys():
    print(i)
for i in d1.values():
    print(i)

for i in d1.items():
    print(i)


d1[1]

d1[3]=['e','f']
d1[1]=['g','h']
d.clear()
d2=d.copy()
d1[4]
d1.get(4)

d2.update(d1)

for i in d2:
    print(i, end=" ")
t=tuple([1,2,3])
t1=(23,)

t2=t+t1
t*2

t.count(1)

t.index(2)

s=set(t)
for i in s:
    print(i)

s.add(7)

#numpy
#numerical python
#enumerate
#map
#zip
#arrays

l=[1,2,3]
l1=['a','b','c']
l2=list(zip(l,l1))
d1=dict(zip(l,l1))

for i, j in enumerate(l1):
    print(i,j)


def sq(x):
    return x**2

for i in map(sq,l):
    print(i)
#numpy
import numpy as np
l1=[[1,2,3],[3,4,5],[5,6,7]]
arr1=np.array(l1)

arr1.shape
a=np.arange(10).astype("float")
a.resize(2,5)
a.size
a[0,0]
a[1,2]











 

