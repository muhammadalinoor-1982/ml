import numpy as np

s = np.array([[[65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77]], [[65,85,79,35,43,69,88,92,56,77], [65,85,79,35,43,69,88,92,56,77], [9,8,7,6,5,4,3,2,1,10]]])
print('Array',s)
print('Length of Array',len(s))
print('Dimention of Array',s.ndim)
print('slic 7 to 1 with 2 step value: ',s[1,2,2:9:2])
