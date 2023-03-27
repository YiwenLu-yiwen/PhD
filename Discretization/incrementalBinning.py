# Using log natural base
from collections import defaultdict
from scipy.stats import entropy
import numpy as np

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
    
class Binning2:

    @staticmethod
    def trivial(x, y):
        n, _ = x.shape
        k = len(np.unique(y))
        bins = np.zeros(n, dtype=int)
        max_bin = 0
        counts =  defaultdict(int) # np.zeros(n, dtype=int)
        counts[0] = n
        y_counts = defaultdict(lambda : np.zeros(k, int)) # np.zeros(shape=(n, k), dtype=int)
        _, y_counts[0] = np.unique(y, return_counts=True)
        cond_entr = defaultdict(float) # np.zeros(n)
        cond_entr[0] = entropy(y_counts[0], base=np.e)
        return Binning2(x, y, bins, max_bin, counts, y_counts, cond_entr, cond_entr[0], 1)
    
    @staticmethod
    def from_assignment(x, y, bins):
        binning = Binning2.trivial(x, y)
        for i, dest in enumerate(bins):
            binning.move(i, dest)
        return binning

    def __init__(self, x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr, non_empty_bin_count):
        n, _ = x.shape
        self.n = n
        self.x = x
        self.y = y
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

    def move(self, i, dest):
        orig = self.bins[i]
        if orig == dest: 
            return
        
        self.bins[i] = dest
        c = self.y[i]
        self.mean_cond_entr = self.mean_cond_entr - self.counts[orig]*self.cond_entr[orig]/self.n - self.counts[dest]*self.cond_entr[dest]/self.n
        self.counts[orig] -= 1
        self.counts[dest] += 1
        self.non_empty_bin_count = self.non_empty_bin_count + (self.counts[dest] == 1) - (self.counts[orig] == 0)
        self.y_counts[orig][c] -= 1
        self.y_counts[dest][c] += 1
        self.cond_entr[orig] = incremental_entropy(self.cond_entr[orig], self.counts[orig]+1, self.y_counts[orig][c]+1, self.y_counts[orig][c])
        self.cond_entr[dest] = incremental_entropy(self.cond_entr[dest], self.counts[dest]-1, self.y_counts[dest][c]-1, self.y_counts[dest][c])
        self.mean_cond_entr = self.mean_cond_entr + self.counts[orig]*self.cond_entr[orig]/self.n + self.counts[dest]*self.cond_entr[dest]/self.n


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

    def best_cut_off(self, order, obj, cutpoint_index=None, criteria=None, dim=None):
        _max_bin = self.max_bin
        split_off_bins = {}
        origins = np.zeros(self.n, dtype=int)
        obj_star = float('inf')
        i_star = -1
        m = 0 if cutpoint_index is not None else self.n-1
        value_lst = []
        # forward
        for i in range(self.n):
            j = order[i]
            b, _b, split_off_bins = self.move_to_cut_off(j, split_off_bins)
            origins[i] = b
            self.move(j, _b)
            obj_value = obj(self)
            if cutpoint_index is not None:
                if len(cutpoint_index) > m and cutpoint_index[m] == i:
                    m += 1
                elif len(cutpoint_index) == m:
                    break
                else:
                    continue
            if criteria in ['bonferroni_duplicate', 'bonferroni_unique', 'orginal']:
                if obj_value < obj_star:
                    i_star, obj_star = i, obj_value
            elif criteria in ['sine_duplicate', 'sine_unique']:
                if obj_value is not None:
                    value_lst.append((obj_value[0], obj_value[1], i, dim))
        
        # rewind
        m = min(self.n-1, i) if cutpoint_index is not None else self.n-1
        for i in range(m, -1, -1):
            j = order[i]
            self.move(j, origins[i])
        self.max_bin = _max_bin
        if criteria in ['bonferroni_duplicate', 'bonferroni_unique', 'orginal']:
            return i_star, obj_star
        elif criteria in ['sine_duplicate', 'sine_unique']:
            return value_lst