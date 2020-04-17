import numpy as np
# print(np.arange(3), 1)
# print([np.arange(3), 1])
# onehot_y[[0,1], [0]] = 1
# print(onehot_y)
p_x = np.array([[1,1],[2,3],[4,2]])
onehot_y = np.array([[0,1],[1,0],[0,1]])


print('sum', np.sum(p_x, axis=0))
print('sum', np.sum(p_x, axis=1))