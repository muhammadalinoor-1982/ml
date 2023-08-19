# ****************************  PANDAS Study **************************

# Link: https://www.youtube.com/playlist?list=PLjVLYmrlmjGdEE2jFpL71LsVH5QjDP5s4

# Data Structure in pandas: 1. Series(1D Data), 2. Data Frame(2D Data), 3. Panel(3D Data)

import pandas as pd

# ******************** 1. Series(1D Data) *****************

# Perform a list using Series() Function
a = [6,7,8,9,10]
b = pd.Series(a)
c = pd.Series(a, index=['x','y','z','x',100], dtype='float', name='aupu') # Change index and data type with change of data type name of list data
print(c)
print(b) # Get data with index
print('Get 5 from this list: ', b[4])
print(type(b)) # Show Data Type of 'b'

# Perform a Dictionary using Series() Function
x = {'name':['python', 'c', 'c++', 'java'], 'rank':[1,4,3,2], 'pop':'good'}
y = pd.Series(x)
print(y)

# Perform Single data as series using Series() Function
z = pd.Series(12)
w = pd.Series(12, index=[9,8,7,6,5,4,3]) # Multiple Index of one value
w1 = pd.Series(12, index=[9,8,7,6]) # Broadcust between w and w1
print(w+w1)
print(z)
print(w)

# ******************** 2. Data Frame(2D Data) *****************

# Perform a list using DataFrame() Function
x = [[9,8,7],[6,5,4]]
y = pd.DataFrame(x)
z = {'a': pd.Series([9,8,7]), 'b': pd.Series([6,5,4])}
w = pd.DataFrame(z)
print('2D List Data Frame: ', '\n', y)
print('Two Series making a Data Frame:  ', '\n', w)

# Perform a Dictionary using DataFrame() Function
x = {'a':[9,8,7], 'b':[6,5,4], 100:500}
y = pd.DataFrame(x)
y1 = pd.DataFrame(x, columns=['b',100], index=['d','g',22]) # If Need specific column and change index 
print(y1)
print(y)
print(y['b'][2]) # Get Specific value of Specific column

