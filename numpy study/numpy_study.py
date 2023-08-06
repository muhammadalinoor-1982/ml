import numpy as np

aray = np.array([1,2,3,4,5,6,7,8,9])
print(aray)

# timeit function [Need to use colab or jupitar notebook]
# timeit function use to execute loop fastly
# %timeit np.arange(1,9)**4

# List to Array Convertion using numpy
l = []
for i in range(1,5):
    n = int(input("Enter Number: " )) 
    l.append(n)
    print(np.array(l))

# How Calcutale 1D, 2D, 3D Array/Matrix of Dimention. Example Below:

# 1D Matrix or Array -> []
oned = np.array([1,2,3])
print(oned)
print("Number of Dimention or Matrix is: ", oned.ndim)

# 2D Matrix or Array -> [[]]
twod = np.array([[1,2,3],[1,2,3],[1,2,3]])
print(twod)
print("Number of Dimention or Matrix is: ", twod.ndim)

# 3D Matrix or Array -> [[[]]]
threed = np.array([[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]])
print(threed)
print("Number of Dimention or Matrix is: ", threed.ndim)

# Multi Dimention or Matrix or Array
ndm = np.array([1,2,3], ndmin = 10)
print(ndm)
print("Number of Dimention or Matrix is: ", ndm.ndim)

# Array/Matrix using zeros() Function

# 4 Columns Matrix using zeros() Function
mz = np.zeros(4)
print(mz)

# 3 Rows and 4 Columns Matrix (3 by 3 Matrix) using zeros() Function
mz = np.zeros((3,4))
print(mz)
print("Number of Dimention or Matrix is: ", mz.dnim)

# Array/Matrix using ones() Function

# 4 Columns Matrix using ones() Function
mo = np.ones(4)
print(mo)

# 3 Rows and 4 Columns Matrix (3 by 3 Matrix) using ones() Function
mo = np.ones((3,4))
print(mo)
print("Number of Dimention or Matrix is: ", mo.dnim)

# Empty Array/Matrix using empty() Function
# Empty Array Contain Previous Data of Memory
me = np.empty((3,4))
print(me)
print("Number of Dimention or Matrix is: ", me.dnim)

# Arrange Array/Matrix using arange() Function
# Arrange Array Contain Secuence of Numbers like 1,2,3,4,5,6,....
ma = np.arange(4)
print(ma)
print("Number of Dimention or Matrix is: ", ma.dnim)

# Digonal Array/Matrix using eye() Function
# Digonal Array/Matrix Contain Secuence combination of 0 and 1 like:
'''
[[1,0,0]
 [0,1,0]
 [0,0,1]]
''' 
md = np.eye(3,3)
print(md)
print("Number of Dimention or Matrix is: ", md.dnim)

# Linearly in a space interval of an Array/Matrix using linspace()
ml = np.linspace(0,20,num=5)
print(ml)
print("Number of Dimention or Matrix is: ", ml.ndim)

# Random Array/Matrix using random.rand() Function to Generate Random value between 0 and 1
mr = np.random.rand(3,3)
print(mr)
print("Number of Dimention or Matrix is: ", mr.dnim)

# Random Array/Matrix using random.randn() Function to Generate Random value and it's close to 0.
# This function generate positive and nagetive numbers randomly 
mrn = np.random.randn(3,3)
print(mrn)
print("Number of Dimention or Matrix is: ", mrn.dnim)

# Random Array/Matrix using random.ranf() Function to Generate Random value and it's very close to 0.
# This function generate numbers between 0.0 to 1.0 randomly 
mrf = np.random.ranf((3,3))
print(mrf)
print("Number of Dimention or Matrix is: ", mrf.dnim)

# Random Array/Matrix using random.randint() Function to Generate Random value with Max, Min and Total Numbers.
# mri = np.random.randint(Max->3, Min->10, Total Numbers of Value->5)  
mri = np.random.randint(3,10,5)
print(mri)
print("Number of Dimention or Matrix is: ", mri.dnim)
