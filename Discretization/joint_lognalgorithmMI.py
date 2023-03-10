import math
from copy import deepcopy
from sklearn.utils import check_array
import numpy as np
from estimators import naive_estimate
from multiprocess import Pool
from explora.information_theory.permutation_model import expected_mutual_information_permutation_model
from early_stopping import match_stopping
from binning import equal_width

def sortValue(subsetData_dic, dim):
    """ sort value in particular dim
    """
    values = subsetData_dic['values']
    subsetData_dic['values'] = values[values[:,dim].argsort()]
    return subsetData_dic

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
        result += -np.log2(p)*p
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
        cnt = 0
        for key in subsetData_dic[each]:
            new_dic['right'][key]['count'] = subsetData_dic[each][key]['count']
            cnt += subsetData_dic[each][key]['count']
        if cnt:
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
    sorted_subsetData_dic['right'][value_y]['min_index'] -= 1
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

def SingleDimension(sorted_data, sort_dim_bin, dim, best_subsetData_list, sample_size, h_y, best_mi, permut, early_stopping, pre_discretized=False, selected_bins=[], num_cutpoints=None, num_selected=None, delta=0.1):
    """For multi-processing use
    Find single dimension comparison, one point at a time.
    Applied chi-square
    sorted_data: sorted data for specific dimension
    dim: specific dimension
    best_subsetData_list: current formated data partipations
    sample_size: total sample sizes
    h_y: H(Y)
    best_mi: current best MI 
    permut: True/False use permutation model
    early_stopping: lazy, chi_square
    num_cutpoints: total number of cutpoints
    num_selected: number of selected cutpoints
    """
    size = len(sorted_data)
    selected_bin = None
    x_discretization, y_values = sampleXY(best_subsetData_list) # create x discretization list
    best_candidate_subsetData_list, best_dim, best_value, degree_of_freedom, new_delta = [], None, None, 0, 0 # set default values
    all_classes = np.unique(y_values)
    # sorted all subset data, complexity O(nlogn)
    for i in range(len(best_subsetData_list)):
        best_subsetData_list[i] = sortValue(best_subsetData_list[i], dim)
    
    candidate_subsetData_list = deepcopy(best_subsetData_list)
    if early_stopping in ['chi_square', 'chi_square_adjust']:
        num_pre_bins = len(candidate_subsetData_list)
    for j in range(len(sorted_data)):
        value_x = sorted_data[j]
        
        if pre_discretized:
            if len(selected_bins) == len(set(sort_dim_bin)):
                break
            if sort_dim_bin[j] in selected_bins:
                continue
        
        for i in range(len(candidate_subsetData_list)): # O(n), select Point complexity is O(1)
            candidate_subsetData_list[i], change_value_list = selectPoint(candidate_subsetData_list[i], dim, value_x)
            
            # add changed index, change_value = (index, sign)
            # modify discretization list
            for change_value in change_value_list:
                x_discretization[change_value[0]] = change_value[1]
        
        candidate_mi = h_y - conditionEntropy(candidate_subsetData_list, sample_size)
        if permut:
            candidate_mi -= expected_mutual_information_permutation_model(x_discretization, y_values, num_threads=1) # use panos library #permutation(x_discretization, y_values).summary()/h_y #
        
        # mutual information is normalized loglikelihood
        if candidate_mi > best_mi:
            best_candidate_subsetData_list = deepcopy(candidate_subsetData_list)
            best_mi = candidate_mi
            best_dim = dim
            best_value = value_x
            selected_bin = sort_dim_bin[j] if pre_discretized else selected_bin
        
            
    return best_candidate_subsetData_list, best_mi, best_dim, best_value, selected_bin

class jointDiscretizationMI:
    """Main algorithm
    data: data matrix
    permut: True if use permutation
    """
    def __init__(self, permut=True, delta=0.05, early_stopping=None, num_cutpoints=None, pre_discretized=False, n_bins=None, bin_method=equal_width) -> None:
        self.permut = permut
        self.delta = delta
        self.early_stopping = early_stopping
        self.num_cutpoints = num_cutpoints
        self.pre_discretized = pre_discretized
        self.n_bins = n_bins
        self.bin_method = bin_method
        self.best_subsetData_list = []
        self.step_mi_list = []
        self.dim_list = []
        self.cutpoint_list = []

    def fit(self, data):
        # initialization
        pool = Pool()
        data = check_array(data)
        all_classes = np.unique(data[:, -1])
        initial_subsetData_dic = reformat_data(data, all_classes)
        best_subsetData_list = reformat_all([initial_subsetData_dic], all_classes)
        sample_size, p = data.shape
        stop = False
        dim_list, best_value_list, best_mi, best_dim, step_mi_list, num_selected = [], [], 0, None, [], 0
        h_y = naive_estimate(data[:, -1]) # naive estimate H(Y)
        num_cutpoints = self.num_cutpoints if self.num_cutpoints else data.shape[0]* (data.shape[1]-1)
        num_cutpoints = (data.shape[1]-1) * math.ceil(np.log2(sample_size)) if self.pre_discretized else num_cutpoints
        loop_times = len(data[0])-1 # predictors loop time
        dims_list = [_ for _ in range(loop_times)]
        sort_data_list = [data[data[:, dim].argsort()][:,dim] for dim in dims_list]
        sample_size_list = [sample_size for _ in range(loop_times)]
        h_y_list = [h_y for _ in range(loop_times)]
        permut_list = [self.permut for _ in range(loop_times)]
        early_stopping_list = [self.early_stopping for _ in range(loop_times)]
        delta_list = [self.delta for _ in range(loop_times)]
        num_cutpoints_list = [num_cutpoints for _ in range(loop_times)]
        pre_discretized_list = [self.pre_discretized for _ in range(loop_times)]
        selected_bins_list = [[] for _ in range(loop_times)]

        n_bins = n_bins if self.n_bins else math.ceil(np.log2(sample_size))
        sort_all_bin = [self.bin_method(sort_data_list[dim], k=n_bins) for dim in dims_list] if self.pre_discretized else [[]for dim in dims_list]
        pre_best_mi, num_pre_bins = 0, 1
        while not stop:
            # initialization
            best_dim, best_candidate_subsetData_list = None, []
            best_subsetData_list_list = [deepcopy(best_subsetData_list) for _ in range(loop_times)]
            best_mi_list = [best_mi for _ in range(loop_times)]
            num_selected_list = [num_selected for _ in range(loop_times)]
            result = pool.starmap(SingleDimension, zip(sort_data_list, sort_all_bin, dims_list, best_subsetData_list_list, sample_size_list, 
                                                        h_y_list, best_mi_list, permut_list, early_stopping_list, pre_discretized_list, selected_bins_list,
                                                        num_cutpoints_list, num_selected_list, delta_list))

            # if not change, we stop the loop
            for each in result:
                if each[1] > best_mi: 
                    if self.early_stopping == 'chi_square' and each[-3] in dim_list: #or len(dim_list)==1:
                        continue
                    else:
                         best_candidate_subsetData_list, best_mi, best_dim, best_value, best_selected_bin = each
            if self.early_stopping == 'chi_square_adjust':
                new_delta = self.delta/(num_cutpoints-num_selected-1)
            elif self.early_stopping == 'chi_square':
                new_delta = self.delta/(sample_size*(p-num_selected-1))

            if best_candidate_subsetData_list:
                best_subsetData_list = reformat_all(best_candidate_subsetData_list, all_classes)
                degree_of_freedom = len(best_subsetData_list) - num_pre_bins
                if match_stopping(current_value=best_mi, previous_value=pre_best_mi, size=sample_size, degree_of_freedom=degree_of_freedom, delta=new_delta, typ=self.early_stopping):
                    pre_best_mi = best_mi
                    num_pre_bins = len(best_subsetData_list)
                    dim_list.append(best_dim)
                    best_value_list.append(best_value)
                    step_mi_list.append(best_mi)
                    num_selected = num_selected + 1
                    selected_bins_list[best_dim].append(best_selected_bin)
                else:
                    stop=True
            else:
                stop=True
            # print(best_dim, best_value, best_mi)
        pool.close()
        self.best_subsetData_list, self.step_mi_list, self.dim_list, self.cutpoint_list = best_subsetData_list, np.array(step_mi_list)/h_y, dim_list, best_value_list
        return self.best_subsetData_list, self.step_mi_list, self.dim_list, self.cutpoint_list, len(self.best_subsetData_list), len(self.dim_list)

def optSingleDimension(sorted_data, sort_dim_bin, dim, best_subsetData_list, sample_size, h_y, best_mi, permut, early_stopping, pre_discretized=False, num_cutpoints=None, num_selected=None, delta=0.1, all_classes=[0,1]):
    """This algorithm is for stagewise
    """
    stop, dim_list, value_list, best_mi_list, selected_bins =False, [], [], [], []
    while not stop:
        single_best_candidate_subsetData_list, single_best_mi, single_best_dim, single_best_value, selected_bin = \
                    SingleDimension(sorted_data, sort_dim_bin, dim, best_subsetData_list, sample_size, h_y, best_mi, permut, early_stopping, pre_discretized, selected_bins, num_cutpoints, num_selected, delta)
        
        if single_best_mi > best_mi:
            best_mi = single_best_mi
            best_subsetData_list = deepcopy(single_best_candidate_subsetData_list)
            best_subsetData_list = reformat_all(best_subsetData_list, all_classes)
            dim_list.append(dim)
            value_list.append(single_best_value)
            best_mi_list.append(best_mi)
            selected_bins.append(selected_bin)
            num_selected += 1
        else:
            stop=True
    return best_subsetData_list, best_mi, value_list, dim_list, best_mi_list, num_selected

class stageWiseDiscretizationMI:
    """Main algorithm
    data: data matrix
    permut: True if use permutation
    """
    def __init__(self, permut=True, delta=0.05, early_stopping=None, num_cutpoints=None, pre_discretized=False, n_bins=None, bin_method=equal_width) -> None:
        self.permut = permut
        self.delta = delta
        self.early_stopping = early_stopping
        self.num_cutpoints = num_cutpoints
        self.pre_discretized = pre_discretized
        self.n_bins = n_bins
        self.bin_method = bin_method
        self.best_subsetData_list = []
        self.step_mi_list = []
        self.dim_list = []
        self.cutpoint_list = []

    def fit(self, data):
        # initialization
        pool = Pool()
        data = check_array(data)
        all_classes = np.unique(data[:, -1])
        initial_subsetData_dic = reformat_data(data, all_classes)
        best_subsetData_list = reformat_all([initial_subsetData_dic], all_classes)
        sample_size = len(data)
        n, p = data.shape
        stop = False
        copy_data = deepcopy(data)
        # permut = permut if not early_stopping else False
        dims_list, best_values_list, best_mi, best_dim, best_dim_list, step_mi_list, num_selected = [_ for _ in range(len(copy_data[0])-1)], [], -np.infty, None, [], [], 0
        h_y = naive_estimate(data[:, -1]) # naive estimate H(Y)
        num_cutpoints = self.num_cutpoints if self.num_cutpoints else data.shape[0]* (data.shape[1]-1)
        num_cutpoints = (data.shape[1]-1) * math.ceil(np.log2(sample_size)) if self.pre_discretized else num_cutpoints

        sort_data_list = [data[data[:, dim].argsort()][:,dim] for dim in dims_list]
        n_bins = n_bins if self.n_bins else math.ceil(np.log2(sample_size))
        sort_all_bin = [self.bin_method(sort_data_list[dim], k=n_bins) for dim in dims_list] if self.pre_discretized else [[]for dim in dims_list]
        pre_best_mi, num_pre_bins = h_y, 1
        while not stop:
            # initialization
            best_dim, best_candidate_subsetData_list = None, []
            loop_times = len(dims_list)
            best_subsetData_list_list = [deepcopy(best_subsetData_list) for _ in range(loop_times)]
            sample_size_list = [sample_size for _ in range(loop_times)]
            h_y_list = [h_y for _ in range(loop_times)]
            best_mi_list = [best_mi for _ in range(loop_times)]
            permut_list = [self.permut for _ in range(loop_times)]
            early_stopping_list = [self.early_stopping for _ in range(loop_times)]
            num_selected_list = [num_selected for _ in range(loop_times)]
            delta_list = [self.delta for _ in range(loop_times)]
            num_cutpoints_list = [num_cutpoints for _ in range(loop_times)]
            pre_discretized_list = [self.pre_discretized for _ in range(loop_times)]
            all_classes_list = [all_classes for _ in range(loop_times)]
            result = pool.starmap(optSingleDimension, zip(sort_data_list, sort_all_bin, dims_list, best_subsetData_list_list, sample_size_list, 
                                                        h_y_list, best_mi_list, permut_list, early_stopping_list, pre_discretized_list,
                                                        num_cutpoints_list, num_selected_list, delta_list, all_classes_list))

            # if not change, we stop the loop
            for each in result:
                if each[1] > best_mi:
                    best_candidate_subsetData_list, best_mi, best_value_list, best_dim, best_mi_sublist, num_selected = each

            if self.early_stopping == 'chi_square_adjust' or self.early_stopping == 'chi_square':
                new_delta = self.delta/(num_cutpoints-num_selected-1)

            if best_candidate_subsetData_list:
                best_subsetData_list = reformat_all(best_candidate_subsetData_list, all_classes)
                degree_of_freedom = len(best_subsetData_list) - num_pre_bins
                if match_stopping(best_mi, pre_best_mi, sample_size, degree_of_freedom=degree_of_freedom, delta=new_delta):
                    pre_best_mi = best_mi
                    num_pre_bins = len(best_subsetData_list)
                    dim_index = dims_list.index(best_dim[0])
                    dims_list.remove(best_dim[0])
                    sort_data_list.pop(dim_index) 
                    best_dim_list += best_dim
                    best_values_list += best_value_list
                    step_mi_list += best_mi_sublist
                    assert num_selected == len(best_values_list), "Not same number of cutpoints in Stagewise"
                else:
                    stop = True
            else:
                stop=True
        pool.close()
        self.best_subsetData_list, self.step_mi_list, self.dim_list, self.cutpoint_list = best_subsetData_list, np.array(step_mi_list)/h_y, best_dim_list, best_value_list
        return self.best_subsetData_list, self.step_mi_list, self.dim_list, self.cutpoint_list, len(self.best_subsetData_list), len(self.dim_list)