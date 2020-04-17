import numpy as np
x = [[0],[1],[2],[3],[4]]
n = 5
n_cluster = 3
first_point = np.random.randint(n)
center = x[first_point]
exist_points = np.array(first_point)
centers = np.array(first_point)
for i in range(n_cluster - 1):
    max_distance = 0
    new_point = 0
    # find largest distance
    for point in range(n):
        if point not in exist_points:
            point1 = x[point]
            point2 = center
            list_len = len(x[point])
            sum = 0
            for j in range(list_len):
                sum += pow(abs(point1[j] - point2[j]), 2)
            distance = pow(sum, 1 / 2)
            if distance > max_distance:
                max_distance = distance
                new_point = point
    # update center
    center = x[new_point]
    exist_points = np.append(exist_points, new_point)
    centers = np.append(centers, new_point)
print ("centers",centers)

x = np.array([[0,0],[1,2],[2,3]])
c = np.array([[0,0],[1,1]])
print ([np.linalg.norm(x-c[k],axis=1)**2 for k in range(2)])
print (np.expand_dims(c, axis=1))
c = np.expand_dims(c, axis=1)
print (np.absolute(x-c))
print (np.linalg.norm(x-c , axis=2)**2)
print (np.argmin (np.linalg.norm(x-c , axis=2), axis=0))
r = np.argmin (np.linalg.norm(x-c , axis=2), axis=0)


p = [True,True,False]
a = np.where(p)
print('a',a)
i = np.array([[7,7],[5,8],[8,9]])
x[a] = i[a]
print(x)
