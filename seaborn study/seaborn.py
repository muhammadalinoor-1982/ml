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

# Histogram Plot with it's perameters with GitHub Dataset
sns.displot(
            df['bill_length_mm'], 
            bins=[34,36,38,40,42,44,46], 
            kde=True, 
            rug=True,
            color='brown',
           ) # log_scale=True
plt.show()

# Bar Plot with GitHub Dataset
sns.barplot(x=df.island, y=df.bill_length_mm)
plt.show()

# Bar Plot with it's perameters with GitHub Dataset
sns.set(style='darkgrid')
sns.barplot(
            x='island', 
            y='bill_length_mm', 
            data=df, 
            hue='sex',
            order=['Dream', 'Torgersen', 'Biscoe'],
            hue_order=['Female', 'Male'],
            #ci=100
            errorbar=('ci', 100),
            n_boot=2,
            orient='v',
            #color='r' 
            palette='icefire',
            saturation=50,
            errcolor='r',
            errwidth=3,
            capsize=0.2,
            dodge=True,
            alpha=1
           )
plt.show()