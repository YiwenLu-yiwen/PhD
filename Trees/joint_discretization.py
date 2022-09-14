from random import sample
from re import S
from tabnanny import check
import numpy as np
from permutation import permutation
from functions import gain, fGain, entropy
from sklearn.utils import check_X_y, check_array
from copy import deepcopy


def split(node, dim, value):
    """For a single node split
    """
    left, right = list(), list()
    for each in node:
        if each[dim] < value:
            left.append(each)
        else:
            right.append(each)
    return left, right

def new_nodes(node_list, dim, value):
    """Create all node list
    """
    new_node_list = []
    for node in node_list:
        left, right = split(node, dim, value)
        if left:
            new_node_list.append(left)
        if right:
            new_node_list.append(right)
    return new_node_list

def info_gain(node_list, y_dict):
    """y_dict = {points, value}
    Information gain
    """
    X, y = list(), list()
    for i in range(len(node_list)):
        for each in node_list[i]:
            X.append(i)
            y.append(y_dict[tuple(each)])
    X, y = np.array(X), np.array(y)
    return fGain(X, y) - permutation(X, y).summary()/entropy(y)

# select one x
def joint_discretization(data, target):
    # main function
    dim_list, final_gain, previous_gain, node_list = [], -np.infty, -np.infty, [deepcopy(data)]
    y_dict = dict(zip([tuple(each) for each in data], target))
    data, target = check_X_y(data, target)
    stop=False
    while not stop:
        for row in data:
            for dim in range(len(row)):
                new_node_list = new_nodes(node_list, dim, row[dim])
                candi_gain = info_gain(new_node_list, y_dict)
                if candi_gain > final_gain:
                    final_gain = candi_gain
                    final_nodes = new_node_list
                    final_dim = dim
                    # final_value = row[dim]
        if previous_gain == final_gain:
            stop=True
        else:
            previous_gain = final_gain
            node_list = final_nodes
            dim_list.append(final_dim)
    return final_gain, dim_list, node_list

if __name__ == '__main__':
    pass