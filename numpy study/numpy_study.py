import numpy as np

# **************** A R R A Y or M A T R I X **********************

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

# ****************** D A T A   T Y P E *********************
dt = np.array([100, 200, 300, 400])
print("\n", "Data Type: ", dt.dtype)

# Convertion of Data Type
dtc = np.array([100, 200, 300, 400], dtype = np.int8)
print("\n", "Data Type: ", dtc.dtype)

# Another Way to Convertion of Data Type
dtf = np.array([100, 200, 300, 400], dtype = 'f')
print("\n", "Data Type: ", dtf.dtype)
print(dtf)

# Change Data Type Using float32() Function
dtft = np.array([100, 200, 300, 400])
new = np.float32(dt)

print("\n", "Data Type: ", dtft.dtype)
print(dtft)

print("\n", "Data Type: ", new.dtype)
print(new)

# Change Data Type Using astype() Function
dtff = np.array([100, 200, 300, 400])
new_1 = dtff.astype(float)

print("\n", "Data Type: ", dtff.dtype)
print(dtff)

print("\n", "Data Type: ", new_1.dtype)
print(new_1)

#******************* ARITHMATIC OPERATION IN AARRAY *****************   

# Useful Arithmatic Function:
# np.add(a,b), np.subtract(a,b), np.multiply(a,b), np.divide(a,b), np.mod(a,b), np.power(a,b), np.reciprocal(1/a) 
# np.min(x), np.max(x), np.argmin(x), np.sqrt(x), np.sin(x), np.cos(x), np.cumsum(x)  

# Addition with intiger
m = np.array([100, 200, 300, 400, 500])
madd = m+3
print(madd)

# Addition with Array
m = np.array([100, 200, 300, 400, 500])
mm = np.array([100, 200, 300, 400, 500])
madd = m+mm
print(madd)

# Arithmatic operation using numpy function
m1 = np.array([100, 200, 300, 400, 500])
m2 = np.array([100, 200, 300, 400, 500])
madd = np.add(m1,m2)
print(madd)

# Addition Using 2D Matrix
m1 = np.array([[100, 200, 300, 400, 500],[100, 200, 300, 400, 500]])
m2 = np.array([[100, 200, 300, 400, 500],[100, 200, 300, 400, 500]])
madd = np.add(m1,m2)
print(madd)

#  Reciprocal function np.reciprocal(a)
m = np.array([[100, 200, 300, 400, 500],[100, 200, 300, 400, 500]])
mm = np.array([[100, 200, 300, 400, 500],[100, 200, 300, 400, 500]])
madd = np.add(m,mm)
ras = np.reciprocal(madd)
print(ras)

# Using np.min(), np.max(), np.argmin(), np.argmax() Function 
m = np.array([9, 6, 8, 7, 34, 6, 2, 1, 4, 11])
print("Min : ", np.min(m), ", Min Position : ", np.argmin(m))
print("Max : ", np.max(m), ", Max Position : ", np.argmax(m))

# Axis in numpy axis=0 indicate row of matrix and axis=1 indicate column of matrix
m = np.array([[9, 6, 8, 7, 34],[6, 2, 1, 4, 11]])
print("Min : ", np.min(m, axis=0))
print("Max : ", np.max(m, axis=1))
print(np.sqrt(m)) # Square Root Fynction

# cumsum function working addition of pervious value with next value in a 1D matrix
m = np.array([9, 6, 8, 7, 34])
print(np.cumsum(m))
# Result: [ 9 15 23 30 64]

# ***************** Shap and Reshap Array **************************

# 2D Array/Matrix using shape, to find the shape of Array 
m = np.array([[9, 6, 8, 7, 34], [9, 6, 8, 7, 34]])
print(m)
print(m.shape)

# Multi Dimentional Array/Matrix using shape, to find the shape of Array
m0 = np.array([[9, 6, 8, 7, 5], [9, 6, 8, 7, 5], [9, 6, 8, 7, 5]], ndmin=6)
print(m0)
print(m0.shape)

# Reshape Array/Matrix using reshape
m = np.array([9, 6, 8, 7, 3, 5])
x = m.reshape(2,3)
print(x)

# Reshape 1D Array/Matrix to 3D then again 3D to 1D
m = np.array([9, 6, 8, 7, 3, 5, 9, 6, 8, 7, 3, 5]) # <<<< 1D
x = m.reshape(2,3,2) # <<<< 3D
y = x.reshape(-1) # <<<< 1D
print(y)

# ***************** Broadcusting **************************

# Rule#1: count from right side, if get 1 between(Array) one of them. Then condition #1 will be justified
# Rule#2: Both of them(Array) maximum value should be same

# Broadcusting Technique: Addition of 1 by 3 Array with 3 by 1 Array
m = np.array([1,2,3]) # 1 by 3 Array
print(m.shape)
n = np.array([[1],[2],[3]]) # 3 by 1 Array
print(n.shape)
o = m + n
print(o)

# Another Example: Addition of 2 by 1 Array with 2 by 3 Array
m = np.array([[1],[2]])
print(m.shape)
n = np.array([[1,2,3], [1,2,3]])
print(n.shape)
o = m + n
print(o)

# ***************** Slicing **************************

# Check Index in an Array
s = np.array([9,8,7,6,5])
print('Index of: ', s[2]) # Index of 7
print('Nagetive Index of: ', s[-3]) # Nagetive Index of 7

# Get The value of Specific Index in a 3D Array
s = np.array([[[9,8], [7,6], [2,9]], [[45,95], [55, 22], [98, 75]]])
print('Dimention of Array is : ', s.ndim) 
print('Get the value of Index [1, -2, -2] is: ', s[1, -2, -2])

# Array Slicing 'start:stop:step'
s = np.array([65,85,79,35,43,69,88,92,56,77])
print('Array',s)
print('Length of Array',len(s))
print('Dimention of Array',s.ndim)
print('slic 79 to 92: ',s[2:8])
print('slic 79 to End: ',s[2:])
print('slic 56 to start: ',s[:8])
print('slic array but after 2 steps later: ',s[::2]) # Mention in the square brecat ony number of step (::2)
print('slic 79 to 56 but after 2 steps later: ',s[2:8:2]) # index of 56 is 8 but n-1 is goes to 7 (n=8 so 8-1=7)
# (::2) First : use for Start, Second : use for End.

# 3D Array Slicing
s = np.array([[[65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77]], [[65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77], [9,8,7,6,5,4,3,2,1,10]]])
print('Array',s)
print('Length of Array',len(s))
print('Dimention of Array',s.ndim)
print('slic 7 to 1 with 2 step value: ',s[1,2,2:9:2])

# ***************** Iteration **************************

# Iteration of 3D Array/Matrix using nditer() function
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
for i in np.nditer(m):
    print(i) 

# Use flags=['buffered'] for extra space and op_dtypes=['S'] to convert string data type
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
for i in np.nditer(m, flags=['buffered'], op_dtypes=['S']):
    print(i)

# To get Index with Data in a 3D Array/Matrix use ndenumerate() function 
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
for i,d in np.ndenumerate(m):
    print(i,d)

# ***************** Basic difference between copy and view **************************
#  For Copy >> If change original Data: Copy can not change change original Data
#  For View >> If change original Data: View can change change original Data

# Example of Copy using copy() function
m = np.array([9,8,7,6,5,4,3,2,1])
c = m.copy()
m[1]=80
print('Original Data: ', m)
print('Copy Data: ', c)
# output:
# Original Data:  [ 9 80  7  6  5  4  3  2  1]
# Copy Data:  [9 8 7 6 5 4 3 2 1]

# Example of View using view() function
m = np.array([9,8,7,6,5,4,3,2,1])
v = m.view()
m[1]=80
print('Original Data: ', m)
print('View Data: ', v)
# output:
# Original Data:  [ 9 80  7  6  5  4  3  2  1]
# View Data:  [9 80 7 6 5 4 3 2 1]


# ***************** Join and Split **************************
# concatenate, stack, array_split

# Join Array using concatenate() function
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
n = np.array([[[11,12,13],[14,15,16],[17,18,19]]])
rw = np.concatenate((m,n), axis=0)
clm = np.concatenate((m,n), axis=1)
rw_clm = np.concatenate((m,n), axis=2)
print('Row Concatenate','\n',rw, '\n')
print('Column Concatenate','\n',clm,'\n')
print('Row and Column Concatenate','\n',rw_clm)

# Join Array using stack() function
# Using 'h' with stack funtion like, hstack() output will be 'Horizontal' Array
# Using 'v' with stack funtion like, vstack() output will be 'Vartical' Array
# Using 'd' with stack funtion like, dstack() output will be 'Hight' Array
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
n = np.array([[[11,12,13],[14,15,16],[17,18,19]]])
horizontal = np.hstack((m,n))
vartical = np.vstack((m,n))
hight = np.dstack((m,n))
print('Horizontal Stack','\n', horizontal , '\n')
print('Vartical Stack','\n', vartical ,'\n')
print('Hight Stack','\n', hight)

# To Splite Multi Dimentional Array using array_split() function
m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
x = np.array_split(m, 3, axis=0)
print(x)