import numpy as np
# from permutation import permutation
from functions import gain
from estimators import naive_estimate
from sklearn.utils import check_X_y
from copy import deepcopy
from early_stopping import chi_square
from multiprocess import Pool

# This algorithm runs very slow

def split(node, dim, value):
    """For a single node split
    """
    left, right = list(), list()
    for each in node:
        if each[dim] <= value:
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
    return gain(X, y)# - permutation(X, y).summary()/entropy(y)

def single_loop(node_list, dim, value, y_dict):
    """For multi-processing only
    """
    new_node_list = new_nodes(node_list, dim, value)
    gains = info_gain(node_list, y_dict)
    return gains, new_node_list, dim, value

def partial_comparison(data, node_list, y_dict, early_stopping, num_pre_bins, pre_gain, pool):
    node_list_list, dim_list, value_list = list(), list(), list()
    final_gain, final_nodes, final_value = None, None, None
    n, p = data.shape
    for dim in range(p):
        uniq_values = np.unique(data[:, dim])
        for value in uniq_values:
            dim_list.append(dim)
            value_list.append(value)
    y_dic_list = [y_dict for _ in range(len(value_list))]
    node_list_list = [node_list for _ in range(len(value_list))]
    new_node_list_list = pool.starmap(new_nodes, zip(node_list_list, dim_list, value_list))
    gains_list = pool.starmap(info_gain, zip(new_node_list_list, y_dic_list))
    final_dim, final_p_value = -1, np.infty
    for i in range(len(gains_list)):
        degree_of_freedom = len(new_node_list_list[i]) - num_pre_bins
        p_current = chi_square(current_value=gains_list[i], previous_value=pre_gain, size=n, degree_of_freedom=degree_of_freedom)
        if p_current < final_p_value:
            final_p_value = p_current
            # if final_dim !=-1 and final_dim < dim_list[i]:
            #     continue
            final_gain = gains_list[i]
            final_nodes = new_node_list_list[i]
            final_dim = dim_list[i]
            final_value = value_list[i]
    return final_gain, final_nodes, final_dim, final_value, final_p_value

class simpleJointDiscretizationMI:

    def __init__(self, early_stopping, delta=0.05) -> None:
        self.early_stopping = early_stopping
        self.delta = delta
        pass

    def fit(self, data, target):
        # main function
        data, target = check_X_y(data, target)
        n, p = data.shape
        pool = Pool()
        dim_list, value_list, final_gain, previous_gain, node_list = [], [], 0, 0, [deepcopy(data)]
        y_dict = dict(zip([tuple(each) for each in data], target))
        data, target = check_X_y(data, target)
        h_y = naive_estimate(target)
        stop=False
        step_mi_list= []
        num_pre_bins = 1
        while not stop:
            final_gain, final_nodes, final_dim, final_value, final_p_value = partial_comparison(data, node_list, y_dict, self.early_stopping, num_pre_bins, previous_gain, pool)
            if self.early_stopping == 'chi_square_adjust':
                new_delta = self.delta/(p*(n-1)-len(value_list)-1)
            elif self.early_stopping == 'chi_square':
                new_delta = self.delta/((n-1)*(p-len(value_list)-1))

            if final_p_value <= new_delta and final_gain!=None:
                previous_gain = final_gain
                num_pre_bins = len(final_nodes)
                node_list = final_nodes
                dim_list.append(final_dim)
                value_list.append(final_value)
                step_mi_list.append(final_gain)
            else:
                stop=True
                break
        pool.close()
        return dim_list, step_mi_list/h_y, num_pre_bins, value_list, num_pre_bins
