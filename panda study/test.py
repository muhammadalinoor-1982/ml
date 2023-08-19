import pandas as pd
x = [[9,8,7],[6,5,4]]
y = pd.DataFrame(x)
z = {'a': pd.Series([9,8,7]), 'b': pd.Series([6,5,4])}
w = pd.DataFrame(z)
print('2D List Data Frame: ', '\n', y)
print('Two Series making a Data Frame:  ', '\n', w)
