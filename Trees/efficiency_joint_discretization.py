import math
from copy import deepcopy
from sklearn.utils import check_array
import numpy as np
from permutation import permutation
from estimators import naive_estimate
from multiprocess import Pool

def sortValue(subsetData_dic, dim):
    """ sort value in particular dim
    """
    values = subsetData_dic['values']
    subsetData_dic['values'] = values[values[:,dim].argsort()]
    return subsetData_dic

def countPoint(sorted_subsetData_dic, value_y):
    """sorted_subsetData_dic = {'values':[], 'index':0, 'sign':0,
    
                                'left' :{0: {'count':0, 'min_index':-1, 'max_index':-1}, 
                                         1: {'count':0, 'min_index':-1, 'max_index':-1}},
                                         
                                'right':{0: {'count':0, 'min_index':-1, 'max_index':-1}, 
                                         1: {'count':0, 'min_index':-1, 'max_index':-1}}}
    index: next index to check value,
    sign: for permutation use, represent what class it belongs to left_class = sign-1, right_class = sign
    """
    # update y_distribution
    sorted_subsetData_dic['left'][value_y]['count'] += 1
    sorted_subsetData_dic['right'][value_y]['count'] -= 1
    
    # update y_values
    # check left index, if not exist, add right min_index
    if sorted_subsetData_dic['left'][value_y]['min_index'] == -1:
        sorted_subsetData_dic['left'][value_y]['min_index'] = sorted_subsetData_dic['right'][value_y]['min_index']
        sorted_subsetData_dic['left'][value_y]['max_index'] = sorted_subsetData_dic['left'][value_y]['min_index']
    # if exist, left max_index + 1, right min_index - 1
    sorted_subsetData_dic['left'][value_y]['max_index'] += 1
    sorted_subsetData_dic['right'][value_y]['min_index'] += 1
    assert sorted_subsetData_dic['right'][value_y]['min_index'] <= sorted_subsetData_dic['right'][value_y]['max_index'], 'Index Error'
    # update index
    sorted_subsetData_dic['index'] += 1
    # get sign
    sign = sorted_subsetData_dic['sign']
    return sorted_subsetData_dic, (sorted_subsetData_dic['left'][value_y]['max_index']-1, sign-1) # (index, sign) need to modify)

def selectPoint(sorted_subsetData_dic, dim, value_x):
    """ select one dimension
    """
    # dim should smaller than len(data[0])-1
    
    # check same dimension
    index = sorted_subsetData_dic['index']
    change_point_list = []
    if index == len(sorted_subsetData_dic['values'][:, dim]) :
        return sorted_subsetData_dic, []
    
    if value_x != sorted_subsetData_dic['values'][:, dim][index]:
        return sorted_subsetData_dic, []

    # check if the same value
    while value_x == sorted_subsetData_dic['values'][:, dim][index]:
        value_y = sorted_subsetData_dic['values'][index, :][-1]
        sorted_subsetData_dic, change_point = countPoint(sorted_subsetData_dic, value_y)
        index = sorted_subsetData_dic['index']
        change_point_list.append(change_point)
        # if match the maximum index, means data has already ends
        if index == len(sorted_subsetData_dic['values'][:, dim]):
            break
    return sorted_subsetData_dic, change_point_list
    
def bin_entropy(dic, sample_size):
    """Get entropy of each bin
        dic: dictionary in each bin
        sample_size: Total sample size
    """
    summation = sum([each['count'] for each in dic.values()])
    result = 0
    for key in dic:
        if not (dic[key]['count'] and summation):
            continue
        p = dic[key]['count']/summation
        result += -math.log2(p)*p
    return result * summation/sample_size
    
def h_yx(sorted_subsetData_dic, sample_size):
    """Get sum of single bin entropy
    """
    result = 0
    for each in ['left', 'right']:
        result += bin_entropy(sorted_subsetData_dic[each], sample_size)
    return result

def conditionEntropy(subsetData_list, sample_size):
    """ Get sum of all bins entropy
    """
    result = 0
    for sorted_subsetData_dic in subsetData_list:
        result += h_yx(sorted_subsetData_dic, sample_size)
    return result
    

def createSample(all_classes):
    """Standard sample dictionary for data structure
    """
    sample_dic = {'values': None, 'index':0, 'sign':0}
    inner_dic = dict(zip(all_classes, [{'count':0, 'min_index':-1, 'max_index':-1} for each in all_classes]))
    for each in ['left', 'right']:
        sample_dic[each] = deepcopy(inner_dic)
    return sample_dic

def reformat_data(sorted_data, all_classes):
    """Transfer data into dictionary
    """
    uniq, cnt = np.unique(sorted_data[:, -1], return_counts=True)
    data_dic = createSample(all_classes)
    data_dic['values'] = sorted_data
    for i in range(len(uniq)):
        data_dic['right'][uniq[i]]['count'] = cnt[i]
    data_dic['sign'] = 2
    return data_dic
    
    
def reformat_dic(subsetData_dic, all_classes):
    """Create new subsetData_dic for new iteration
    """
    new_list = [] # store new list
    index = subsetData_dic['index']
    
    if index == 0: # if not creating new index, return orginal one
        return [subsetData_dic]
    
    for each in ['left', 'right']:
        # check data exist or not
        if len(subsetData_dic['values'])==0:
            continue

        new_dic = createSample(all_classes)
        if each == 'left': # left value is small
            new_dic['values'] = subsetData_dic['values'][:index]
        else: # right value
            new_dic['values'] = subsetData_dic['values'][index:]
        
        # only update counts
        for key in subsetData_dic[each]:
            new_dic['right'][key]['count'] = subsetData_dic[each][key]['count']
            
        new_list.append(new_dic)
    return new_list

def updateIndex(subsetData_dic, latest_index):
    """Modify the index of each bin
    """
    for key in subsetData_dic['right']:
        if subsetData_dic['right'][key]['count']:
            subsetData_dic['right'][key]['min_index'] = latest_index
            max_index = latest_index + subsetData_dic['right'][key]['count']
            subsetData_dic['right'][key]['max_index'] = max_index
            latest_index = max_index
    return subsetData_dic, latest_index
        
def reformat_all(subsetData_list, all_classes):
    """Combine reformat dictionary and create in all list
    """
    result_list = []
    sign = 1
    for subsetData_dic in subsetData_list:
        sublist = reformat_dic(subsetData_dic, all_classes) # add sign for each list
        for j in range(len(sublist)):
            sublist[j]['sign'] = 2*sign
            sign += 1
            result_list.append(deepcopy(sublist[j])) # add deepcopy to avoid potential issue 
    
    # update index
    latest_index = 0 
    for i in range(len(result_list)):
        result_list[i], latest_index = updateIndex(result_list[i], latest_index)
    return result_list

def sampleXY(subsetData_list):
    """Get X,Y class list
    """
    x_list, y_list = [], []
    for each_dic in subsetData_list:
        sign = each_dic['sign']
        for key in each_dic['right']:
            subx_list = [sign for _ in range(each_dic['right'][key]['min_index'], each_dic['right'][key]['max_index'])]
            x_list += subx_list

            suby_list = [key for _ in range(each_dic['right'][key]['min_index'], each_dic['right'][key]['max_index'])]
            y_list += suby_list
    return np.array(x_list), np.array(y_list)

def singleDimension(data, dim, best_subsetData_list, sample_size, h_y, best_fmi, permut):
    """For multi-processing use
    Single dimension comparison
    """
    sorted_data = data[data[:, dim].argsort()][:,dim] # only sorted related dimension, save space complexity  
    x_discretization, y_values = sampleXY(best_subsetData_list) # create x discretization list
    best_candidate_subsetData_list, best_dim, best_value = [], None, None
    # sorted all subset data, complexity O(nlogn)
    for i in range(len(best_subsetData_list)):
        best_subsetData_list[i] = sortValue(best_subsetData_list[i], dim)
    
    candidate_subsetData_list = deepcopy(best_subsetData_list)
    for j in range(len(sorted_data)):
        value_x = sorted_data[j]
        
        for i in range(len(candidate_subsetData_list)): # O(n), select Point complexity is O(1)
            candidate_subsetData_list[i], change_value_list = selectPoint(candidate_subsetData_list[i], dim, value_x)
            
            # add changed index, change_value = (index, sign)
            # modify discretization list
            for change_value in change_value_list:
                x_discretization[change_value[0]] = change_value[1]
        
        candidate_fmi = 1 - conditionEntropy(candidate_subsetData_list, sample_size)/h_y 
        if permut:
            candidate_fmi -= permutation(x_discretization, y_values).summary()
        # comparison
        if candidate_fmi > best_fmi:
            best_candidate_subsetData_list = deepcopy(candidate_subsetData_list)
            best_fmi = candidate_fmi
            best_dim = dim
            best_value = value_x
    return best_candidate_subsetData_list, best_fmi, best_dim, best_value
    
def jointDiscretization(data, permut=True):
    """Main algorithm
    data: data matrix
    permut: True if use permutation
    """
    # initialization
    pool = Pool()
    data = check_array(data)
    all_classes = np.unique(data[:, -1])
    initial_subsetData_dic = reformat_data(data, all_classes)
    best_subsetData_list = reformat_all([initial_subsetData_dic], all_classes)
    sample_size = len(data)
    stop = False
    dim_list, best_value_list, best_fmi, best_dim = [], [], -np.infty, None
    h_y = naive_estimate(data[:, -1]) # naive estimate H(Y)
    while not stop:
        # initialization
        best_dim, best_candidate_subsetData_list = None, []
        rep = len(data[0])-1 # repeat time
        data_list = [data for _ in range(rep)]
        dims_list = [_ for _ in range(rep)]
        best_subsetData_list_list = [deepcopy(best_subsetData_list) for _ in range(rep)]
        sample_size_list = [sample_size for _ in range(rep)]
        h_y_list = [h_y for _ in range(rep)]
        best_fmi_list = [best_fmi for _ in range(rep)]
        permut_list = [permut for _ in range(rep)]
        result = pool.starmap(singleDimension, zip(data_list, dims_list, best_subsetData_list_list, sample_size_list, h_y_list, best_fmi_list, permut_list))

        # if not change, we stop the loop
        for each in result:
            if each[1] > best_fmi:
                best_candidate_subsetData_list = each[0]
                best_fmi = each[1]
                best_dim = each[2]
                best_value = each[3]
        if best_candidate_subsetData_list:
            best_subsetData_list = reformat_all(best_candidate_subsetData_list, all_classes)
            dim_list.append(best_dim)
            best_value_list.append(best_value)
        else:
            stop=True
    pool.close()
    return best_subsetData_list, best_fmi, dim_list, best_value_list