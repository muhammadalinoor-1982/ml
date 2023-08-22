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

# ******************** Arithmatic Operation: Add, Subs, Mult, Div, Condition *****************

# Addition Operation
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4]})
z['Addition'] = z['a'] + z['b']
print(z)

# Apply Condition in DataFrame
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4]})
z['comp'] = z['a'] <= 8
z['comp.'] = z['b'] >= 6
print(z)

# ******************** Insert and Delete *****************

# Insertion using insert() Function
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4]})
z.insert(2,'new', z['a']) # Insert a column. Column name: new, column index: 2, value of new column: z['a']
z.insert(3,'new1', [99,88,77]) # Insert a column. Column name: new1, column index: 3, value of new column:[99,88,77]
z['new2'] = z['new1'][:1] # Insert a column with limited value [:1] Here 1 is index of new1 column
print(z)

# Delete using pop() or del() Function
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4], 'c':[11,12,13]})
z1 = z.pop('c') # using pop()
print(z1)
print(z)

z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4], 'c':[11,12,13]})
del z['b'] # using del()
print(z)

# ******************** Create CSV File *****************

# Create CSV File using to_csv() Function
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4], 'c':[11,12,13]})
z.to_csv('create_csv_File.csv')
z.to_csv('create_csv_File1.csv', index=False) # Create csv file without index
z.to_csv('create_csv_File2.csv', index=False, header=['ID','Name','Roll']) # Create csv file without index and with Header
print(z)

# ******************** Read CSV File **********************
# Use Google Colab for this Operation
# First of all Connect with Google Colab to import files
# Then Upload local machine file into google colab
#**********************************************************

# Upload file and read data form csv file from google colab
import pandas as pd

from google.colab import files
upload_csv_file = files.upload()

df = pd.read_csv("electronic-card-transactions.csv")
df # Show all csv file data

df = pd.read_csv("electronic-card-transactions.csv", nrows=5)
df # How many rows what to get

df = pd.read_csv("electronic-card-transactions.csv", usecols=['SYMBOL','CHANGE'])
df # Get Specific Column

df = pd.read_csv("electronic-card-transactions.csv", usecols=[0,4])
df # Get Specific Column using index number of column

df = pd.read_csv("electronic-card-transactions.csv", skiprows=[15,21])
df # Skip Rows

df = pd.read_csv("electronic-card-transactions.csv", index_col=['SYMBOL'])
df # If need any column replaced with index

df = pd.read_csv("electronic-card-transactions.csv", header = 5)
df # If need any row replaced with header

df = pd.read_csv("electronic-card-transactions.csv", names = ['col1', 'col2', 'col3', 'col4', 'col5'])
df # If need change header name

df = pd.read_csv("electronic-card-transactions.csv", header = None, prefix='col')
df # If header name change with prefix

df = pd.read_csv("electronic-card-transactions.csv", dtype = {'CHANGE':'float'})
df # Change data type of specific column

# ******************** Some Function of CSV File **********************

df = pd.read_csv("electronic-card-transactions.csv")
df.head() # Read data from your uploaded csv file from first 5 row of data

df.tail() # Read data from your uploaded csv file from last 5 row of data

df.index # Get the information about index. Example: RangeIndex(start=0, stop=29, step=1)

df.columns # Get to know about columns

df.describe() # Get to know about min, max, 25%, 50%, 75%, std, mean of data

df[19:26] # Get specific range of data (Slicing)

df.index.array # Get index as array with it's length and type

df.to_numpy() # Get Data as numpy array

# Another way to Get Data as numpy array
import numpy as np
data = np.asarray(df)
data

df.sort_index(axis = 0, ascending = False)  # Get data/Index as descending order

df['SYMBOL'][1]='aupu'
df  # Change data with warning coz it is not right process

df.loc[2, 'SYMBOL'] = 'noor'
df # Change data without warning coz it is right process

df.loc[[21,22,23,24,25],['SECURITY', 'CHANGE']] # Get Specific data from dataset

df.loc[21:25,['SECURITY', 'CHANGE']] # Another way to get specific data from dataset

df.iloc[27,3] # Another way to get specific data from dataset. here 27 is row of index and 3 is column of index

df.drop(6, axis=0) # Delete 6th row of dataset. here axis=0 is indicate row axis and 1 is column axis

df.drop('SYMBOL', axis=1) # Delete SYMBOL column of dataset. here axis=1 is indicate column axis

