import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = []  # the possible unique values of the feature to be split

        self.child_class = []

    # TODO: try to split current node
    def split(self):
        max_score = 0
        best_child_list = []
        best_label_list = []
        best_class_list = []
        temp_split = -1
        if len(self.features[0]) == len(self.feature_uniq_split):
            self.splittable = False
            return
        else:
            # find best feature to split
            # best_feature = self.features[0]

            # transpose
            transpose_feature = np.transpose(self.features)
            # feature index
            for feature_index in range(len(self.features[0])):
                can_split = True
                for already_split in self.feature_uniq_split:
                    if feature_index == already_split:
                        can_split = False
                if can_split:
                    # print('feature_index',len(self.features[0]))
                    temp_child_list = []
                    temp_label_list = []
                    temp_count_list = []
                    temp_class_list = []
                    # feature class
                    for feature_class in np.unique(transpose_feature[feature_index]):
                        child_num = np.unique(transpose_feature[feature_index]).size
                        temp_child = []
                        temp_label = []
                        temp_count = []
                        temp_class = feature_class
                        # feature number
                        for feature_num in range(len(self.features)):
                            if self.features[feature_num][feature_index] == feature_class:
                                temp_child.append(self.features[feature_num])
                                temp_label.append(self.labels[feature_num])

                        # count label number and calculate IG
                        for label_class in np.unique(self.labels):
                            count_temp_label_num = 0
                            for temp_label_index in range(len(temp_label)):
                                if temp_label[temp_label_index] == label_class:
                                    count_temp_label_num += 1
                            temp_count.append(count_temp_label_num)
                        # print('temp_count', temp_count)
                        temp_child_list.append(temp_child)
                        temp_label_list.append(temp_label)
                        temp_count_list.append(temp_count)
                        temp_class_list.append(temp_class)
                    parentlist = []
                    # parent_score = 0
                    for child in range(len(temp_count_list[0])):
                        num = 0
                        for index in range(len(temp_count_list)):
                            num += temp_count_list[index][child]
                        parentlist.append(num)
                    # count_parent = sum(parentlist)
                    parent_score = -1 * Util.Information_Gain(0, [parentlist])
                    # for j in range(len(parentlist)):
                    #     if parentlist[j] != 0:
                    #         parent_score -= (parentlist[j] / count_parent) * (np.log2((parentlist[j] / count_parent)))
                    score = Util.Information_Gain(parent_score, temp_count_list)
                    if score < 0.000000000000001:
                        score = 0.0
                    print(score>0.0)
                    print('score', score, temp_label_list)
                    if score > max_score:
                        max_score = score
                        best_child_list = temp_child_list
                        best_label_list = temp_label_list
                        best_class_list = temp_class_list
                        temp_split = feature_index
                    elif score == max_score and score > 0:
                        if len(temp_label_list) > len(best_label_list):
                            best_child_list = temp_child_list
                            best_label_list = temp_label_list
                            best_class_list = temp_class_list
                            temp_split = feature_index
                            # self.dim_split = feature_index
            # self.dim_split = temp_split
            if max_score == 0:
                raise NotImplementedError
                self.splittable = False
                self.cls_max = self.labels[0]
                return
            print('max_score', max_score, best_label_list)
            self.dim_split = temp_split
            self.feature_uniq_split.append(temp_split)
            self.child_class = best_class_list
            # end for loop

            for child_index in range(len(best_label_list)):
                child_num_cls = np.unique(best_label_list[child_index]).size
                if len(best_child_list[child_index]) == 0:
                    return
                else:
                    child_node = TreeNode(best_child_list[child_index], best_label_list[child_index], child_num_cls)
                    child_node.feature_uniq_split = self.feature_uniq_split
                if child_node.splittable:
                    child_node.split()
                self.children.append(child_node)
            return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            for index in range(len(self.child_class)):
                if feature[self.dim_split] == self.child_class[index]:
                    return self.children[index].predict(feature)
        return self.cls_max

''' version 1->19
        if len(self.features[0]) == len(self.feature_uniq_split):
            self.splittable = False
            return

        else:
            # find best feature to split
            # best_feature = self.features[0]
            max_score = -999999
            best_child_list = []
            best_label_list = []
            temp_split = 0
            #transpose
            transpose_feature = np.transpose(self.features)
            # feature index
            for feature_index in range(len(self.features[0])):
                can_split = True
                for already_split in self.feature_uniq_split:
                    if feature_index == already_split:
                        can_split = False
                if can_split:
                    temp_child_list = []
                    temp_label_list = []
                    temp_count_list = []

                    # print('feature_index',len(self.features[0]))
                    #feature class
                    for feature_class in np.unique(transpose_feature[feature_index]):
                        child_num = np.unique(transpose_feature[feature_index]).size
                        temp_child = []
                        temp_label = []
                        temp_count = []
                        # feature number
                        for feature_num in range(len(self.features)):
                            if self.features[feature_num][feature_index] == feature_class:
                                temp_child.append(self.features[feature_num])
                                temp_label.append(self.labels[feature_num])

                        # count label number and calculate IG
                        for label_class in np.unique(self.labels):
                            count_temp_label_num = 0
                            for temp_label_index in range(len(temp_label)):
                                if temp_label[temp_label_index] == label_class:
                                    count_temp_label_num += 1
                            temp_count.append(count_temp_label_num)
                        # print('temp_count', temp_count)
                        temp_child_list.append(temp_child)
                        temp_label_list.append(temp_label)
                        temp_count_list.append(temp_count)

                    score = Util.Information_Gain(0, temp_count_list)
                    if score > max_score:
                        max_score = score
                        best_child_list = temp_child_list
                        best_label_list = temp_label_list
                        temp_split = feature_index
            self.dim_split = temp_split
            self.feature_uniq_split.append(temp_split)
            # end for loop
            for child_index in range(len(best_label_list)):
                child_num_cls = np.unique(best_label_list).size
                child_node = TreeNode(best_child_list[child_index], best_label_list[child_index], child_num_cls)
                child_node.feature_uniq_split = self.feature_uniq_split
                if child_node.splittable:
                    child_node.split()
                self.children.append(child_node)
            return     '''
