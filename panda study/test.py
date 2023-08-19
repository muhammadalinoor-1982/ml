import pandas as pd
z = pd.DataFrame({'a':[9,8,7], 'b': [6,5,4], 'c':[11,12,13]})
z.to_csv('create_csv_File.csv')
z.to_csv('create_csv_File1.csv', index=False) # Create csv file without index
z.to_csv('create_csv_File2.csv', index=False, header=['ID','Name','Roll']) # Create csv file without index and with Header
print(z)
