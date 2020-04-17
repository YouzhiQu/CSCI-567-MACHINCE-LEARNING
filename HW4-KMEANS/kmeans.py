import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    # centers = []
    # center_one = generator.randint(0, n)
    # # centers.append(x[center_one].tolist())
    # centers.append(center_one)
    # for i in range(n_cluster - 1):
    #     distance_sq = [0] * n
    #     for c in range(len(centers)):
    #         for index in range(n):
    #             if index not in centers:
    #                 distance = np.linalg.norm(x[index] - centers[c], ord=2) ** 2
    #                 if (distance_sq[index] != 0 and distance < distance_sq[index]) or distance_sq[index] == 0:
    #                     distance_sq[index] = distance
    #     center_id = distance_sq.index(max(distance_sq))
    #     # centers.append(x[center_id].tolist())
    #     centers.append(center_id)
    first_point = generator.randint(n)
    centers = []
    centers.append(first_point)
    for i in range(n_cluster - 1):
        max_distance = 0
        new_point = 0
        # find largest distance
        for point in range(n):
            if point not in centers:
                valid_distance = pow(10, 10)
                for c in range(len(centers)):
                    point1 = x[point]
                    point2 = centers[c]
                    temp_distance = np.linalg.norm(point1 - point2, ord=2) ** 2
                    if temp_distance < valid_distance:
                        valid_distance = temp_distance
                if valid_distance > max_distance:
                    max_distance = valid_distance
                    new_point = point
        # update center
        centers.append(new_point)
    # centers = centers.tolist()



    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # centroids = x[self.centers]
        # y = np.zeros(N)
        # j = pow(10, 10)
        # iter = 0
        # while iter < self.max_iter:
        #     # assignments
        #     l2_sqare = [np.linalg.norm(x - centroids[k], axis=1, ord=2) ** 2 for k in range(self.n_cluster)]
        #     y = np.argmin(l2_sqare, axis=0)
        #     jnew = np.sum([np.linalg.norm(x[y == k] - centroids[k], ord=2) ** 2 for k in range(self.n_cluster)]) / N
        #     if abs(j - jnew) <= self.e:
        #         break
        #     j = jnew
        #     # update centers
        #     new_centroids = np.array([np.mean(x[y == k], axis=0) for k in range(self.n_cluster)])
        #     index = np.where(np.isnan(new_centroids))
        #     new_centroids[index] = centroids[index]
        #     centroids = new_centroids
        #     iter += 1
        # self.max_iter = iter
        # def computeJ(x, center, y):
        #     return np.sum([np.sum((x[y == k] - center[k]) ** 2) for k in range(self.n_cluster)]) / N
        y = np.zeros(len(x))
        centroids = x[np.copy(self.centers)]
        max_round = self.max_iter
        j = pow(10, 10)
        iter = 0
        while max_round > 0:
            y = np.argmin(np.array([np.linalg.norm(x - centroids[k], axis=1, ord=2) ** 2 for k in range(self.n_cluster)]), axis=0)
            #y = np.argmin(np.linalg.norm(x - np.expand_dims(centroids, axis=1), axis=2), axis=0)
            new_j = np.sum([np.sum((x[y == k] - centroids[k]) ** 2) for k in range(self.n_cluster)]) / N
            #new_j = computeJ(x, centroids, y)
            if np.absolute(j - new_j) <= self.e:
                break
            j = new_j
            centroids_new = np.array([np.mean(x[y == k], axis=0) for k in range(self.n_cluster)])
            pos = np.where(np.isnan(centroids_new))
            centroids_new[pos] = centroids[pos]
            centroids = centroids_new
            max_round -= 1
            iter += 1
        self.max_iter = iter
            # for i in range(len(x)):
            #     min_distance = 9999
            #     for j in range(self.n_cluster):
            #         distance = np.linalg.norm(x[i] - centroids[j], ord=2) ** 2
            #         if distance < min_distance:
            #             min_distance = distance
            #             y[i] = j
            # invalid_update = False
            # center = np.zeros(N, D)
            # for k in range(self.n_cluster):
            #     count = 0
            #     for l in range(len(x)):
            #         if y[l] == k:
            #             center[k] += x[l]
            #             count += 1
            #     if count == 0:
            #         center[k] /= count
            #     else:
            #         invalid_update = True
            #
            # if not invalid_update:
            #     if np.array_equal(centroids, center):
            #         iter += 1
            #         break
            #     else:
            #         centroids = center

        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        def most_common(lst):
            return max(set(lst), key=lst.count)
        k_means = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, iteration = k_means.fit(x, centroid_func)
        count = [y[membership == k].tolist() for k in range(self.n_cluster)]
        centroid_labels = np.array([most_common(count[k]) for k in range(len(count))])

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        y = np.argmin(np.array([np.linalg.norm(x - self.centroids[k], axis=1, ord=2) ** 2 for k in range(self.n_cluster)]), axis=0)
        labels = self.centroid_labels[y]
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    num_center = code_vectors.shape[0]
    img_y = np.argmin(np.array([np.linalg.norm(image - code_vectors[k], axis=2, ord=2) for k in range(num_center)]), axis=0)
    new_im = code_vectors[img_y]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

