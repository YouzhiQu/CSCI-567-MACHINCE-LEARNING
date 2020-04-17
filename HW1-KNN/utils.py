import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    if len(real_labels) != len(predicted_labels):
        raise NotImplementedError
    list_len = len(real_labels)
    count_tp = 0
    count_real = 0
    count_predict = 0
    for i in range(list_len):
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            count_tp += 1
        if real_labels[i] == 1:
            count_real += 1
        if predicted_labels[i] == 1:
            count_predict += 1

    score = (2 * count_tp) / (count_real + count_predict)
    return score


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if len(point1) != len(point2):
            raise NotImplementedError
        list_len = len(point1)
        sum = 0
        for i in range(list_len):
            sum += pow(abs(point1[i] - point2[i]), 3)
        distance = pow(sum, 1/3)
        return distance


    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if len(point1) != len(point2):
            raise NotImplementedError
        list_len = len(point1)
        sum = 0
        for i in range(list_len):
            sum += pow(abs(point1[i] - point2[i]), 2)
        distance = pow(sum, 1/2)
        return distance

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        if len(point1) != len(point2):
            raise NotImplementedError
        list_len = len(point1)
        distance = 0
        for i in range(list_len):
            distance += (point1[i] * point2[i])
        return distance

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if len(point1) != len(point2):
            raise NotImplementedError
        list_len = len(point1)
        csa = 0
        csb = 0
        csc = 0
        for i in range(list_len):
            csa += (point1[i] * point2[i])
            csb += pow(point1[i], 2)
            csc += pow(point2[i], 2)
        csb = pow(csb, 1/2)
        csc = pow(csc, 1/2)
        distance = 1 - csa / (csb * csc)
        return distance

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if len(point1) != len(point2):
            raise NotImplementedError
        list_len = len(point1)
        gka = 0
        for i in range(list_len):
            gka += pow((point1[i] - point2[i]), 2)
        gka = -1/2 * gka
        distance = - np.exp(gka)
        return distance


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        max_score = 0
        for i in distance_funcs:
            for k in range(1, 30, 2):
                cur_model = KNN(k, distance_funcs[i])
                cur_model.train(x_train, y_train)
                cur_score = f1_score(y_val, cur_model.predict(x_val))
                if cur_score > max_score:
                    max_score = cur_score
                    self.best_k = k
                    self.best_distance_function = i
                    self.best_model = cur_model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        max_score = 0
        for j in scaling_classes:
            cur_scaler = scaling_classes[j]()
            after_x_train = cur_scaler(x_train)
            after_x_val = cur_scaler(x_val)
            for i in distance_funcs:
                for k in range(1, min(len(after_x_train), 30), 2):
                        cur_model = KNN(k, distance_funcs[i])
                        cur_model.train(after_x_train, y_train)
                        cur_score = f1_score(y_val, cur_model.predict(after_x_val))
                        if cur_score > max_score:
                            max_score = cur_score
                            self.best_k = k
                            self.best_distance_function = i
                            self.best_scaler = j
                            self.best_model = cur_model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalize_result = []
        for i in range(len(features)):
            temp = 0
            after = []
            for j in range(len(features[i])):
                temp += pow(features[i][j], 2)
            temp = pow(temp, 1/2)
            for j in range(len(features[i])):
                if temp == 0:
                    after.append(0)
                else:
                    after.append(features[i][j]/temp)
            normalize_result.append(after)
        return normalize_result


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    feature_min = None
    feature_max = None

    def __init__(self):
        self.first_time = True

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if self.first_time:
            self.first_time = False
            self.feature_min = [0] * len(features[0])
            self.feature_max = [0] * len(features[0])
            minmax_result = np.zeros([len(features), len(features[0])])
            for i in range(len(features[0])):
                self.feature_max[i] = features[0][i]
                self.feature_min[i] = features[0][i]
                for j in range(len(features)):
                    if features[j][i] > self.feature_max[i]:
                        self.feature_max[i] = features[j][i]
                    if features[j][i] < self.feature_min[i]:
                        self.feature_min[i] = features[j][i]

                for j in range(len(features)):
                    if (self.feature_max[i] - self.feature_min[i]) != 0:
                        minmax_result[j][i] = (features[j][i] - self.feature_min[i]) / (self.feature_max[i] - self.feature_min[i])
            return minmax_result
        else:
            minmax_result = np.zeros([len(features), len(features[0])])
            for i in range(len(features[0])):
                for j in range(len(features)):
                    if (self.feature_max[i] - self.feature_min[i]) != 0:
                        minmax_result[j][i] = (features[j][i] - self.feature_min[i]) / (self.feature_max[i] - self.feature_min[i])
            return minmax_result
#           if self.first_time:
#             self.first_time = False
#             self.feature_min = [999999] * len(features[0])
#             self.feature_max = [-999999] * len(features[0])
#             minmax_result = np.zeros([len(features), len(features[0])])
#             for i in range(len(features[0])):
#                 for j in range(len(features)):
#                     if features[j][i] > self.feature_max[i]:
#                         self.feature_max[i] = features[j][i]
#                     if features[j][i] < self.feature_min[i]:
#                         self.feature_min[i] = features[j][i]
#                 for j in range(len(features)):
#                     minmax_result[j][i] = (features[j][i] - self.feature_min[i]) / (self.feature_max[i] - self.feature_min[i])
#             return minmax_result
#         else:
#             minmax_result = np.zeros([len(features), len(features[0])])
#             for i in range(len(features[0])):
#                 for j in range(len(features)):
#                     minmax_result[j][i] = (features[j][i] - self.feature_min[i]) / (self.feature_max[i] - self.feature_min[i])
#             return minmax_result


a = [1, 1, 1, 1, 0]
b = [1, 0, 1, 1, 0]
features = [[2, -1], [-1, 5], [0, 0]]
print(f1_score(a, b))
result = []
feature_min = [99999999] * len(features[0])
feature_max = [-999999999] * len(features[0])
minmax_result = np.zeros([len(features), len(features[0])])
for i in range(len(features[0])):
    for j in range(len(features)):
        if features[j][i] > feature_max[i]:
            feature_max[i] = features[j][i]
        if features[j][i] < feature_min[i]:
            feature_min[i] = features[j][i]
    for j in range(len(features)):
        minmax_result[j][i] = (features[j][i] - feature_min[i]) / (feature_max[i] - feature_min[i])
print(minmax_result)