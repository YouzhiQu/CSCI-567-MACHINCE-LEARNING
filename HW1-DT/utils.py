import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):

    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    count_temp = [0] * len(branches)
    cur_entropy = [0] * len(branches)
    information_gain = S
    for i in range(len(branches)):
        count_temp[i] = sum(branches[i])
        for j in range(len(branches[i])):
            if branches[i][j] != 0:
                cur_entropy[i] -= (branches[i][j] / count_temp[i]) * (np.log2((branches[i][j] / count_temp[i])))
                # print(cur_entropy[i])
    count_all = sum(count_temp)
    temp_score = 0
    for k in range(len(branches)):
        temp_score += (count_temp[k] / count_all) * cur_entropy[k]
    information_gain -= temp_score
    # if information_gain < 0.0000000000000001:
    #     information_gain = 0
    return information_gain

# TODO: implement reduced error prunning function, pruning your tree on this function


def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    # calculate improve score of each split point, prune this point and loop

    def calculate_score(node, X_test, y_test):
        # calculate this node
        correct_num = 0
        prune_correct_num = 0
        for index in range(len(X_test)):
            predict = root_node.predict(X_test[index])
            if predict == y_test[index]:
                correct_num += 1
            if node.cls_max == y_test[index]:
                prune_correct_num += 1
        score = prune_correct_num - correct_num
        return score

    def prune_point(node, X_test, y_test):
        if not node.splittable:
            return 0, node

        temp_score = calculate_score(node, X_test, y_test)
        temp_node = None
        if temp_score > 0:
            temp_node = node
            # node.splittable = False
            # node.dim_split = None
            # node.children = []
            # node.child_class = []

        for index in range(len(node.child_class)):
            new_feature = []
            new_label = []
            for j in range(len(X_test)):
                if X_test[j][node.dim_split] == node.child_class[index]:
                    new_feature.append(X_test[j])
                    new_label.append(y_test[j])
            child_score, child_new_node = prune_point(node.children[index],new_feature, new_label)
            if child_score > temp_score:
                temp_score = child_score
                temp_node = child_new_node

        if temp_score > 0 and temp_node is not None:
            # temp_node.splittable = False
            # temp_node.dim_split = None
            # temp_node.children = []
            # temp_node.child_class = []
            return temp_score, temp_node
        else:
            return 0, temp_node
    root_node = decisionTree.root_node

    score, prune_node = prune_point(root_node, X_test, y_test)
    while score > 0 and prune_node is not None:
        prune_node.splittable = False
        prune_node.dim_split = None
        prune_node.children = []
        prune_node.child_class = []
        score, prune_node = prune_point(root_node, X_test, y_test)

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


# branches = [[2,5,0],[10,0,3]]
# branches = [[1, 0], [1, 1], [0, 1]]
# count_temp = [0] * len(branches)
# cur_entropy = [0] * len(branches)
# information_gain = 0
# for i in range(len(branches)):
#     count_temp[i] = sum(branches[i])
#     print(count_temp[i])
#     for j in range(len(branches[i])):
#         if branches[i][j] != 0:
#             cur_entropy[i] -= (branches[i][j] / count_temp[i]) * (np.log2((branches[i][j] / count_temp[i])))
# count_all = sum(count_temp)
# print(count_all)
# for k in range(len(branches)):
#     print((count_temp[k] / count_all))
#     information_gain -= (count_temp[k] / count_all) * cur_entropy[k]
# print(information_gain)