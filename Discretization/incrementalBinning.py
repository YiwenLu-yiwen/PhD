# Using log natural base
from collections import defaultdict
from scipy.stats import entropy
from copy import deepcopy
import numpy as np

from explora.information_theory.permutation_model import expected_mutual_information_permutation_model

def convert_to_x_y(y_counts):
    xx, yy = [], []
    for each in y_counts:
        xx += [each] * sum(y_counts[each])
        for i in range(len(y_counts[each])):
            yy += [i] * y_counts[each][i]
    return np.array(xx), np.array(yy)

def entropy_from_counts(counts):
    n = sum(counts)
    return np.log(n) - sum(counts*np.log(counts))/n

def incremental_entropy(h_old, n, c_old, c_new):
    alpha = c_new - c_old
    if n == 0 or n == -alpha: # old or new histogram empty
        return 0.0
    else:
        new_term = c_new*np.log(c_new) if c_new > 0 else 0
        old_term = c_old*np.log(c_old) if c_old > 0 else 0
        return np.log(n+alpha)-(new_term + n*(np.log(n)-h_old) - old_term)/(n+alpha) 

def convert_y(y):
    """Convert y to 0, 1, 2, ...
    """
    uniq_y, _n = np.unique(y, return_counts=True)
    converted_y_dict = dict(zip(np.unique(uniq_y), [_ for _ in range(len(np.unique(uniq_y)))]))    
    converted_y = []
    for each in y:
        converted_y.append(converted_y_dict[each])
    return converted_y

class Binning2:

    @staticmethod
    def trivial(x, y, reliable=False): # modify y
        n, _ = x.shape
        y = convert_y(y) # update y here
        k = len(np.unique(y))
        bins = np.zeros(n, dtype=int)
        max_bin = 0
        counts =  defaultdict(int) # np.zeros(n, dtype=int)
        counts[0] = n
        y_counts = defaultdict(lambda : np.zeros(k, int)) # np.zeros(shape=(n, k), dtype=int)
        _, y_counts[0] = np.unique(y, return_counts=True)
        cond_entr = defaultdict(float) # np.zeros(n)
        cond_entr[0] = entropy(y_counts[0], base=2)
        if reliable:
            xx, yy = convert_to_x_y(y_counts)
            _mo_old = expected_mutual_information_permutation_model(xx, yy, num_threads=1)
            cond_entr[0] += _mo_old
        return Binning2(x, y, bins, max_bin, counts, y_counts, cond_entr, cond_entr[0], 1, reliable)
    
    @staticmethod
    def from_assignment(x, y, bins, reliable=False):
        binning = Binning2.trivial(x, y, reliable)
        for i, dest in enumerate(bins):
            binning.move(i, dest)
        return binning

    def __init__(self, x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr, non_empty_bin_count, reliable=False):
        n, _ = x.shape
        self.n = n
        self.x = x
        self.y = convert_y(y)
        self.bins = bins
        self.max_bin = max_bin
        self.counts = counts
        self.y_counts = y_counts
        self.cond_entr = cond_entr
        self.mean_cond_entr = mean_cond_entr
        self.non_empty_bin_count = non_empty_bin_count
        self.old_mean_cond_entr = mean_cond_entr
        self.old_non_empty_bin_count = non_empty_bin_count
        self.k_ = len(np.unique(y))
        self.reliable = reliable

    def move(self, i, dest):
        orig = self.bins[i]
        if orig == dest: 
            return
        self.bins[i] = dest
        c = self.y[i]
        if self.reliable:
            xx, yy = convert_to_x_y(self.y_counts)
            _mo_old = expected_mutual_information_permutation_model(xx, yy, num_threads=1)
        self.mean_cond_entr = self.mean_cond_entr - self.counts[orig]*self.cond_entr[orig]/self.n - self.counts[dest]*self.cond_entr[dest]/self.n
        self.counts[orig] -= 1
        self.counts[dest] += 1
        self.non_empty_bin_count = self.non_empty_bin_count + (self.counts[dest] == 1) - (self.counts[orig] == 0)
        self.y_counts[orig][c] -= 1
        self.y_counts[dest][c] += 1
        self.cond_entr[orig] = incremental_entropy(self.cond_entr[orig], self.counts[orig]+1, self.y_counts[orig][c]+1, self.y_counts[orig][c])
        self.cond_entr[dest] = incremental_entropy(self.cond_entr[dest], self.counts[dest]-1, self.y_counts[dest][c]-1, self.y_counts[dest][c])
        self.mean_cond_entr = self.mean_cond_entr + self.counts[orig]*self.cond_entr[orig]/self.n + self.counts[dest]*self.cond_entr[dest]/self.n
        if self.reliable:
            xx, yy = convert_to_x_y(self.y_counts)
            _mo_new = expected_mutual_information_permutation_model(xx, yy, num_threads=1)
            self.mean_cond_entr += _mo_new - _mo_old
        
    def move_to_cut_off(self, j, split_off_bins):
        """move real data index j to new bin
        """
        b = self.bins[j]
        if b not in split_off_bins:
            self.max_bin += 1
            split_off_bins[b] = self.max_bin
        _b = split_off_bins[b]
        return b, _b, split_off_bins
    
    def apply_cut_off(self, l, order):
        split_off_bins = {}
        for i in range(l+1):
            j = order[i]
            _, _b, split_off_bins = self.move_to_cut_off(j, split_off_bins)
            self.move(j, _b)

    def best_cut_off(self, order, obj, cutpoint_index_list=[], dim=0):
        _max_bin = self.max_bin
        split_off_bins = {}
        origins = np.zeros(self.n, dtype=int)
        obj_star = float('inf')
        i_star = -1
        m = 0
        y_counts = deepcopy(self.y_counts)
        if len(cutpoint_index_list) == 0:
            return i_star, obj_star, y_counts
        cutpoint_indx_list = [order[i] for i in cutpoint_index_list]
        cutpoint_value_list = np.unique(self.x[:, dim][cutpoint_indx_list])
        # forward
        # for i in range(self.n):
        i = 0
        while i < self.n:
            ###### if next value is the same with previous value, then move to the same bin as previous one ######
            while True:
                j = order[i]
                b, _b, split_off_bins = self.move_to_cut_off(j, split_off_bins)
                origins[i] = b
                self.move(j, _b)
                obj_value = obj(self)
                i += 1
                if i < self.n and self.x[:, dim][order[i-1]] == self.x[:, dim][order[i]]:
                    continue
                else:
                    break
            if len(cutpoint_value_list) > m and cutpoint_value_list[m] == self.x[:, dim][order[i-1]]:
                if obj_value < obj_star:
                    i_star, obj_star, y_counts = i-1, obj_value, deepcopy(self.y_counts)
                m += 1
                if len(cutpoint_value_list) == m:
                    break
            else:
                continue

        # rewind
        m = min(self.n-1, i-1)
        for i in range(m, -1, -1):
            j = order[i]
            self.move(j, origins[i])
        self.max_bin = _max_bin
        return i_star, obj_star, y_counts
    
class BinningOneDim(Binning2):

    @staticmethod
    def trivial(x, y, reliable=True): # modify y
        n, _ = x.shape
        y = convert_y(y) # update y here
        k = len(np.unique(y))
        bins = np.zeros(n, dtype=int)
        max_bin = 0
        counts =  defaultdict(int) # np.zeros(n, dtype=int)
        counts[0] = n
        y_counts = defaultdict(lambda : np.zeros(k, int)) # np.zeros(shape=(n, k), dtype=int)
        _, y_counts[0] = np.unique(y, return_counts=True)
        cond_entr = defaultdict(float) # np.zeros(n)
        cond_entr[0] = entropy(y_counts[0], base=2)
        if reliable:
            xx, yy = convert_to_x_y(y_counts)
            _mo_old = expected_mutual_information_permutation_model(xx, yy, num_threads=1)
            cond_entr[0] += _mo_old
        return BinningOneDim(x, y, bins, max_bin, counts, y_counts, cond_entr, cond_entr[0], 1, reliable)
    
    @staticmethod
    def from_assignment(x, y, bins, reliable=True):
        binning = BinningOneDim.trivial(x, y, reliable)
        for i, dest in enumerate(bins):
            binning.move(i, dest)
        return binning
    
    def __init__(self, x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr, non_empty_bin_count, reliable=True):
        super().__init__(x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr, non_empty_bin_count, reliable)


    def best_cut_off_finely(self, order, obj, cutpoint_index_list=[], dim=0):
        best_obj_star = float('inf')
        best_y_counts = deepcopy(self.y_counts)
        dim_p_list = []
        binning = deepcopy(self)
        while True:
            cond_entr_old = binning.mean_cond_entr
            df_old = binning.non_empty_bin_count
            i_star, obj_star, y_counts = binning.best_cut_off(order, obj, cutpoint_index_list, dim)
            if i_star == -1 or obj_star == float('inf') or obj_star >= best_obj_star: # or abs(obj_star - best_obj_star) < 1e-10
                self = deepcopy(binning)
                return dim_p_list, best_obj_star, self
            binning.apply_cut_off(i_star, order)
            cond_entr_new = binning.mean_cond_entr
            df_new = binning.non_empty_bin_count
            dim_p_list.append((dim, binning.x[order[i_star], dim], obj_star, cond_entr_old-cond_entr_new, df_new))
            cutpoint_index_list = [i for i in cutpoint_index_list if i_star != i]
            best_obj_star, best_y_counts = obj_star, y_counts

        
        
