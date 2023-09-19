# Source Link: https://www.youtube.com/playlist?list=PLjVLYmrlmjGfhqSO3rF4n02rrj9w2Ch2C

# Line Plot with it's perameters
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

a = [1,2,3,4,5,6,7,8,9]
b = [9,8,7,6,5,4,3,2,1]
c = pd.DataFrame({'a':a, 'b':b})

sns.lineplot(x='a', y='b', data=c )

plt.show()

# Line Plot using gitHub Dataset
#https://github.com/mwaskom/seaborn-data
ghd = sns.load_dataset('penguins').head(100)
ghd

sns.lineplot(
             x='bill_length_mm',
             y='bill_depth_mm', 
             data=ghd, 
             hue='sex', 
             size=100, 
             style='sex', 
             palette='Accent', 
             markers=['*', 'o'],
             dashes=True,
             legend='full'
            )

plt.grid()
plt.title('Penguins', fontsize=15)

plt.show()