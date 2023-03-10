import numpy as np
# from permutation import permutation
from functions import gain
from sklearn.utils import check_X_y
from copy import deepcopy
from multiprocess import Pool
from early_stopping import match_stopping

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

def partial_comparison(data, node_list, y_dict, pool):
    node_list_list, dim_list, value_list = list(), list(), list()
    for dim in range(len(data[0])):
        uniq_values = np.unique(data[:, dim])
        for value in uniq_values:
            dim_list.append(dim)
            value_list.append(value)

    y_dic_list = [y_dict for _ in range(len(value_list))]
    node_list_list = [node_list for _ in range(len(value_list))]
    new_node_list_list = pool.starmap(new_nodes, zip(node_list_list, dim_list, value_list))
    gains_list = pool.starmap(info_gain, zip(new_node_list_list, y_dic_list))
    final_gain, final_dim = -1, -1
    for i in range(len(gains_list)):
        if gains_list[i] > final_gain:
            # if final_dim !=-1 and final_dim < dim_list[i]:
            #     continue
            final_gain = gains_list[i]
            final_nodes = new_node_list_list[i]
            final_dim = dim_list[i]
            final_value = value_list[i]    
    return final_gain, final_nodes, final_dim, final_value

class simpleJointDiscretizationPvalue:

    def __init__(self, early_stopping, delta=0.05) -> None:
        self.early_stopping = early_stopping
        self.delta = delta
        pass

    def fit(self, data, target):
        # main function
        data, target = check_X_y(data, target)
        n, p = data.shape
        pool = Pool()
        dim_list, value_list, final_gain, previous_gain, node_list = [], [], -np.infty, 0, [deepcopy(data)]
        y_dict = dict(zip([tuple(each) for each in data], target))
        data, target = check_X_y(data, target)
        stop=False
        step_mi_list= []
        num_pre_bins = 1
        while not stop:
            final_gain, final_nodes, final_dim, final_value = partial_comparison(data, node_list, y_dict, pool)
            
            if self.early_stopping == 'chi_square_adjust':
                new_delta = self.delta/(n*p-len(value_list)-1)
            elif self.early_stopping == 'chi_square':
                new_delta = self.delta/(n*(p-len(value_list)-1))
            degree_of_freedom = len(final_nodes) - num_pre_bins
            if match_stopping(current_value=final_gain, previous_value=previous_gain, size=n, degree_of_freedom=degree_of_freedom, delta=new_delta, typ=self.early_stopping):
                previous_gain = final_gain
                node_list = final_nodes
                dim_list.append(final_dim)
                value_list.append(final_value)
                step_mi_list.append(final_gain)
            else:
                stop=True
                break
        pool.close()
        return final_gain, dim_list, node_list, value_list, step_mi_list