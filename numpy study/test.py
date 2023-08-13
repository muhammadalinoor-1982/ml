import numpy as np

m = np.array([[[9,8,7],[6,5,4],[3,2,1]]])
x = np.array_split(m, 3, axis=0)
print(x)




