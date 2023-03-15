from skimage import io

img = io.imread('arr_num.jpg')
io.imshow(img)

import numpy as np
#creating array
a=np.arange(10).astype("float")
a.resize(2,5)
#Create an array of evenlyspaced values 
#(number of samples)
b=np.linspace(0,2,9)
b.resize(3,3)

a = np.array([1,2,3], )

b = np.array([(4,2,3), (7,5,6)], dtype = float)
c = np.array([[(1.5,2,3), (4,5,6)],[(3,2,1), (4,5,6)]], dtype = float)
#to check the dimensions
c.ndim

#Initial Placeholders 
#Create an array of zeros
z=np.zeros((4,4)) 
#Create an array of ones
o=np.ones((2,2))
#creating a matrix with a predefined value
f=np.full((3,5),'a')
#identity
e=np.eye(4)
r=np.random.random((2,5))
n=np.random.normal(0, 1, (3,3))
a.mean()
a.std()
a.sum()
a.prod()
a.cumsum()
a.cumprod()
x1 = np.random.randint(10, size=6) #one dimension
x2 = np.random.randint(10, size=(3,4)) #two dimension
x3 = np.random.randint(10, size=(3,4,5)) #three dimension
x1.dtype
x1.itemsize
x1.size
x1[-1]
x2[1,1]
x2[:2,:2]
x2[::-1, ::-1]


x2_sub_copy = x2[:2, :2].copy()



x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z=np.concatenate([x, y])
y2=np.array([[1, 2, 3],[4, 5, 6]])
z3=np.concatenate([y2,y2], axis=1)

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 6])
x1=np.arange(16).reshape((4, 4))
u,l=np.vsplit(x1, [2])
le,r=np.hsplit(x1, [2])
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)
print("x % 2 =", x % 2)
np.add(x,5)
xs=np.subtract(x,2)
np.abs(xs)

np.multiply(xs,.6)
np.divide(xs,5)
np.floor_divide(x,2)
np.power(xs,2)
x>2
np.count_nonzero(x>2)
np.sum(x<2)
np.any(x>2)
np.all(x>2)
np.median(x)







