import numpy as np
from scipy.stats import chi2
from multiprocess import Pool
from incrementalBinning import Binning2, BinningOneDim
import pandas as pd
from copy import deepcopy
# from numba import njit

def cond_entr_obj(binning):
    return binning.mean_cond_entr

def chi_square_obj(binning):
    """i means selected points in i-1 iteration
    """
    increased_df = binning.non_empty_bin_count-binning.old_non_empty_bin_count
    return chi2.sf(2*binning.n*(binning.old_mean_cond_entr-binning.mean_cond_entr), increased_df*(binning.k_-1))

# @njit
# def chi_square_cond_entr_obj(binning):
#     """Get chi square and conditional entropy
#     """
#     return (chi_square_obj(binning), cond_entr_obj(binning))

def create_cutpoint_index_obj(_n, gamma, dims):
    """Generate cutpoint index for each variable constantly
    """
    bins = int(_n ** gamma) if gamma <= 1 else gamma
    cut_index, i, j = [], 0, 1
    line = pd.cut(np.arange(_n), bins=bins+1, labels=False)
    while i < _n-1:
        if line[j] != line[i]:
            cut_index.append(i)
        i += 1
        j += 1
    return [cut_index] * dims

class VariableSelectionOracle:

    def __init__(self, base='mi', gamma=10, multi_var=True, reliable=False, verbose=False):
        """
            base: We apply two different methods to compare cutpoints.
                mi: use pearson naive mutual information
                p_value: use chi square hypothesis test value
            criteria: We don't apply any criteria until we have selected all variables or we don't have enough bins.
        """
        self.base = base
        self.gamma = gamma
        self.multi_var = multi_var
        self.reliable = reliable
        self.verbose = verbose
    
    def fit(self, x, y, rr_vars=None):
        self.x = x
        binning = Binning2.trivial(x, y, reliable=self.reliable)
        binning.dim_point_value_ = []
        orders = np.argsort(x, axis=0)
        self.n_, self.p_ = x.shape
        self.num_cuts_selected_ = np.zeros(self.p_, dtype=int)
        dims_ = np.arange(self.p_)
        cutpoint_index_list = create_cutpoint_index_obj(self.n_, self.gamma, self.p_)
        if self.base=='p_value':
            obj = chi_square_obj
        elif self.base=='mi':
            obj = cond_entr_obj
        while True:
            j_star, i_star, obj_star = -1, -1, float('inf')
            pool=Pool()
            res = pool.starmap(binning.best_cut_off, zip([orders[:, _] for _ in dims_], 
                                                         [obj for _ in range(len(dims_))], 
                                                         cutpoint_index_list,
                                                         [dim for dim in dims_]))
            pool.close()
            for j in range(len(dims_)):
                if res[j][1] < obj_star:
                    j_star, i_star, obj_star = dims_[j], res[j][0], res[j][1]
                
            # stop algorithm if not selecting anything
            if i_star == -1 or j_star == -1:
                break
            # if p_value is 1 or mi is 0, stop algorithm
            if (self.base=='p_value' and obj_star == 1) or (self.base=='mi' and obj_star == 0):
                break
            cond_entr_old = binning.mean_cond_entr
            df_old = binning.non_empty_bin_count
            binning.apply_cut_off(i_star, orders[:, j_star])
            if binning.dim_point_value_ == []:
                binning.dim_point_value_.append((-1, None, float('inf'), cond_entr_old, None))
            cond_entr_new = binning.mean_cond_entr
            df_new = binning.non_empty_bin_count
            if df_old == df_new or cond_entr_old <= cond_entr_new: # determine reliable estimation
                break
            # drop dims if we don't have enough cutpoints
            self.num_cuts_selected_[j_star] += 1
            binning.old_mean_cond_entr = cond_entr_new
            binning.old_non_empty_bin_count = df_new
            if self.verbose:
                print(binning.dim_point_value_)
            binning.dim_point_value_.append((j_star, x[orders[i_star, j_star], j_star], obj_star, cond_entr_old-cond_entr_new, df_new-df_old))
            cutpoint_index_list[j_star] = \
                [each for each in cutpoint_index_list[j_star] if each != i_star]
            if self.multi_var == False:
                cutpoint_index_list[j_star] = []
            # add stop signs
            if rr_vars is not None and rr_vars != []:
                if (self.num_cuts_selected_[rr_vars] !=0).all() == True:
                    break
        self.selected_ = np.flatnonzero(self.num_cuts_selected_)
        self.num_cuts_selected_ = self.num_cuts_selected_[self.selected_]
        self.binning = binning
        return self, binning

    def transform(self, x, y):
        return x[:, self.selected_], y
    
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)
    
    def feature_importance_score(self):
        self.n_, self.p_ = self.x.shape
        importance_score = np.zeros(self.p_)
        for each in self.binning.dim_point_value_[1:]:
            j_star, _value = each[0], each[-2]
            importance_score[j_star] += _value
        return importance_score
    
    def order_feature_rank(self):
        features = []
        for each in self.binning.dim_point_value_[1:]:
            if each[0] not in features:
                features.append(each[0])
        for each in np.arange(self.p_):
            if each not in features:
                features.append(each)
        return features

    def score_feature_rank(self):
        return np.argsort(self.feature_importance_score())[::-1]
    
    def get_cutpoints(self):
        cutpoints = []
        for each in self.binning.dim_point_value_[1:]:
            cutpoints.append((each[0], each[1]))
        return cutpoints
    
    def apply_cutpoints(self, dim, values):
        self.x[:, dim] = np.digitize(self.x[:, dim], values)
        return self.x[:, dim]
    


class VariableSelectionReliable:

    def __init__(self, base='mi', gamma=10, verbose=False):
        """
            base: We apply two different methods to compare cutpoints.
                mi: use pearson naive mutual information
                p_value: use chi square hypothesis test value
            criteria: We don't apply any criteria until we have selected all variables or we don't have enough bins.
        """
        self.base = base
        self.gamma = gamma
        self.reliable = True
        self.verbose = verbose
    
    def fit(self, x, y, rr_vars=None):
        self.x = x
        binning = BinningOneDim.trivial(x, y, reliable=self.reliable)
        binning.dim_point_value_ = []
        orders = np.argsort(x, axis=0)
        self.n_, self.p_ = x.shape
        self.num_cuts_selected_ = np.zeros(self.p_, dtype=int)
        dims_ = np.arange(self.p_)
        cutpoint_index_list = create_cutpoint_index_obj(self.n_, self.gamma, self.p_)
        if self.base=='p_value':
            obj = chi_square_obj
        elif self.base=='mi':
            obj = cond_entr_obj
        obj_star = float('inf')
        while True:
            j_star, dim_point_list = -1, []
            pool = Pool()
            res = pool.starmap(binning.best_cut_off_finely, zip([orders[:, _] for _ in dims_], 
                                                        [obj for _ in range(len(dims_))], 
                                                        cutpoint_index_list,
                                                        [dim for dim in dims_]))
            pool.close()
            for j in range(len(dims_)):
                if res[j][1] < obj_star:
                    j_star, dim_point_list, obj_star, binning = j, res[j][0], res[j][1], deepcopy(res[j][2])
            # stop algorithm if not selecting anything
            if dim_point_list == []:
                break
            # if p_value is 1 or mi is 0, stop algorithm
            if (self.base=='p_value' and obj_star == 1) or (self.base=='mi' and obj_star == 0):
                break
            cond_entr_old = binning.mean_cond_entr
            self.num_cuts_selected_[j_star] += 1
            if binning.dim_point_value_ == []:
                binning.dim_point_value_.append((-1, None, float('inf'), cond_entr_old, None))
            if self.verbose:
                print(binning.dim_point_value_)
            binning.dim_point_value_ += dim_point_list
            cutpoint_index_list[j_star] = []
            if rr_vars is not None and rr_vars != []:
                if (self.num_cuts_selected_[rr_vars] !=0).all() == True:
                    break
        self.selected_ = np.flatnonzero(self.num_cuts_selected_)
        self.num_cuts_selected_ = self.num_cuts_selected_[self.selected_]
        self.binning = binning
        return self, binning

    def transform(self, x, y):
        return x[:, self.selected_], y
    
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)
    
    def feature_importance_score(self):
        self.n_, self.p_ = self.x.shape
        importance_score = np.zeros(self.p_)
        for each in self.binning.dim_point_value_[1:]:
            j_star, _value = each[0], each[-2]
            importance_score[j_star] += _value
        return importance_score
    
    def order_feature_rank(self):
        features = []
        for each in self.binning.dim_point_value_[1:]:
            if each[0] not in features:
                features.append(each[0])
        for each in np.arange(self.p_):
            if each not in features:
                features.append(each)
        return features

    def score_feature_rank(self):
        return np.argsort(self.feature_importance_score())[::-1]
    
    def get_cutpoints(self):
        cutpoints = []
        for each in self.binning.dim_point_value_[1:]:
            cutpoints.append((each[0], each[1]))
        return cutpoints
    
    def apply_cutpoints(self, dim, values):
        self.x[:, dim] = np.digitize(self.x[:, dim], values)
        return self.x[:, dim]

