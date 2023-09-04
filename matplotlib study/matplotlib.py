# ****************************  Matplotlib Study **************************

# Link: https://www.youtube.com/playlist?list=PLjVLYmrlmjGcC0B_FP3bkJ-JIPkV5GuZR

# *********** Plot: linear, scatter, bar, stem, step, hist, box, pie, fill between/area plot  ***********

# Install Jupyter Note Book: Go to Python Installation Folder then Go to Script Folder 
# Then Type 'cmd' in the Address bar. Then Type "pip install jupyter" in command promt
# Open Jupyter Notebook: Type "jupyter notebook" in the same command promt
# Open a file in the jupyter notebook: Go to New and Select Notebook then Select python3 (Kernel)
# Then Type "pip install matplotlib" in command promt
# Use matplotlib in 2 ways: 'from matplotlib import pyplot as plt' OR 'import matplotlib.pyplot as plt' 

import matplotlib.pyplot as plt
# Show symple Linear plot using plot() and show() Function
x = [1,3,5,7,9]
y = [2,4,6,8,10]
plt.plot(x,y)
plt.show()

# Show symple bar plot using bar() Function with color attribute 
x = [1,3,5,7,9]
y = [2,4,6,8,10]
c = ['r', 'b', 'g', 'y', 'r']
plt.bar(x,y, color=c)
plt.show()

# Show symple bar plot using bar() Function with verious perameter 
import matplotlib.pyplot as plt
import numpy as np

x = ['lichi', 'painapple', 'banana', 'mango', 'orange', 'watermelon']
y = [75, 25, 35, 80, 60, 55]
w = [78, 35, 30, 85, 65, 50]
z = ['orange', 'brown', 'r', 'b', 'black', 'g']

width=0.1
p = np.arange(len(x))
p1 = [j + width for j in p] # convert x list to p array 

plt.xlabel('Frutes', fontsize=20)
plt.ylabel('popularity')
plt.title('Fruites Statistics')

plt.bar(p,y, width=0.1, color=z, align='edge', edgecolor='y', linewidth=2, linestyle=':', alpha=0.99, label='AA') # for horizontal bar use barh
plt.bar(p1,w, width=0.1, color=z, align='edge', edgecolor='r', linewidth=2, linestyle=':', alpha=0.50, label='BB')

plt.xticks(p+width/2, x, rotation=20) # Show the name of x value
plt.legend()
plt.show()

# Show scatter plot using scatter() Function with verious perameter 
x = [5,1,6,4,9,4,2,2,3,4]
y = [9,8,3,3,1,6,1,4,1,8]
z = [6,4,2,6,4,2,4,3,4,8]
size = [99,89,39,39,19,69,19,49,19,89]
color = [46,98,24,42,63,51,34,85,74,98]

plt.scatter(x,y, c=color, s=size, marker='^', cmap='viridis')
plt.scatter(z,y, color='b')

t=plt.colorbar()
t.set_label('Color Bar')
plt.title('scatter plot', fontsize=20)
plt.xlabel('day')
plt.ylabel('number')

plt.show()

# Show Histogram plot using hist() Function with verious perameter 
no = [10, 39, 28, 26, 35, 55, 21, 39, 27, 16, 25, 15, 24, 51, 28, 15, 18, 54, 15, 45, 48, 48, 56, 29,
 55, 48, 46, 51, 48, 58, 54, 33, 37, 55, 45, 32, 24, 39, 33, 40, 55, 52, 50, 42, 25, 58, 45, 10,
 46, 10]

l = [10,20,30,40,50,60] # Use for bins=l, perameter

plt.hist(no, 'auto', (0, 100), edgecolor='black', cumulative=-1, bottom=10, align='mid', histtype='barstacked', orientation='vertical', rwidth=0.8, log=True, label='Test')

plt.title('Histogram plot', fontsize=20)
plt.xlabel('day')
plt.ylabel('number')
plt.axvline(25, color='g', label='Green Line')
plt.legend()
plt.grid()

plt.show()