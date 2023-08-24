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

# ********************** dropna() and fillna() Function *************************

df.dropna() # Drop Nan value from csv file

df.dropna(axis=1) # Drop Nan value of column from csv file

df.dropna(axis=0) # Drop Nan value of row from csv file

df.dropna(how = 'any') # Drop fully Nan value of row from csv file

df.dropna(how = 'all') # Drop only which row that all column are Nan from csv file

df.dropna(subset = ['CHANGE']) # Drop specific column's Nan value from csv file

df.dropna(inplace = True)
df # Drop all Nan value and create a new dataset

df.dropna(thresh = 2) # Drop single Nan value from csv file

df.fillna('noor') # Fill Nan value with peramiter in csv file

df.fillna({'CHANGE':1000, 'AVG VOLUME':'putin'}) # Specific column's Nan value change with specific data

df.fillna(method = 'ffill') # Fill Nan value with forward value (Fill Pervious row's data of Nan row )

df.fillna(method = 'bfill') # Fill Nan value with backward value (Fill Under row's data of Nan row )

df.fillna(method = 'ffill', axis=1) # Fill Nan value with next side column value 

df.fillna(method = 'bfill', axis=1) # Fill Nan value with pervious side column value

df.fillna('TTTT', inplace = True)
df # Fill all Nan value with peramiter 'TTTT' and create a new dataset

df.fillna('noor', limit = 2) # First 2 Nan value of CSV file, Fill with 'noor'

# ********************** Handling Missing Values Using replace() and interpolate() Function *******************

df.replace(to_replace='Total income', value='noor') # Replace 'Total income' with 'noor' using replace() Function

df.replace([2462.5, 2493.37, 80078.0], 100000) # Replace [2462.5, 2493.37, 80078.0] with 100000 using replace() 

df.replace('[A-Za-z]', 'X', regex=True) # Replace '[A-Za-z]' with 'X' using regex=True

df.replace({'SYMBOL': '[A-Z]'}, 100, regex=True) # Replace Alphabetic Column to Numeric Column using Dictionary

df.replace('anything', method='ffill') # Replace 'anything' using method peramiter

df.replace('anything', method='ffill', limit=3) # Replace 'anything' using method and limit peramiter

df.replace('anything', method='ffill', limit=3, implace=True) # Replace 'anything' using method, limit and implace peramiter
# Note: implace peramiter chnage original data. Implace peramiter do not copy original Data *******************

# Interpolate Function work only numerical not with string *****************
df.interpolate() # Change Nan Value accrose forward value

df.interpolate(method='linear') # Change Nan Value linearly accrose forward value

df.interpolate(method='linear', axis=0) # All data in dataset must be numerical. Otherwise axis peramiter not work

df.interpolate(limit=3) # Nan Value in all the column, change only 3 row of deta all of them   

df.interpolate(limit_direction='forward', limit=3) # Change Nan Value with limit_direction='forward/backword/both'

df.interpolate(limit_direction='forward', limit=3, inplace=True) # Change with Original data using inplace=True

df.interpolate(limit_area='inside') # Change Nan Value with limit_area='inside/outside'


