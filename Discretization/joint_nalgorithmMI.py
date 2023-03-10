import numpy as np
import numpy as np
# from scipy.stats import entropy
from copy import deepcopy
from multiprocess import Pool
from early_stopping import match_stopping

def entropy_from_counts(counts):
    n = np.sum(counts)
    if n==0: return 0
    counts = np.array([each for each in counts if each])
    return np.log2(n) - np.sum(counts*np.log2(counts))/n

def incremental_entropy(h_old, n, c_old, c_new):
    delta = c_new - c_old
    if n == 0 or n == -delta: # old or new histogram empty
        return 0.0
    else:
        new_term = c_new*np.log2(c_new) if c_new > 0 else 0
        old_term = c_old*np.log2(c_old) if c_old > 0 else 0
        return np.log2(n+delta)-(new_term + n*(np.log2(n)-h_old) - old_term)/(n+delta)

class Binning:

    @staticmethod
    def trivial(x, y):
        n, _ = x.shape
        k = len(np.unique(y))
        bins = np.zeros(n+1, dtype=int)
        max_bin = 0
        counts = np.zeros(n+1, dtype=int)
        y_counts = np.zeros(shape=(n+1, k), dtype=int)
        _, y_counts[0, :] = np.unique(y, return_counts=True)
        cond_entr = np.zeros(n+1)
        # cond_entr[0] = entropy(y_counts[0], axis=1, base=2)
        return Binning(x, y, bins, max_bin, counts, y_counts, cond_entr, cond_entr)
    
    @staticmethod
    def from_assignment(x, y, bins):
        n, p = x.shape
        k = len(np.unique(y))
        max_bin = np.unique(bins)[-1]
        counts = np.zeros(n+1, dtype=int)
        _, counts[:max_bin+1] = np.unique(bins, return_counts=True)
        y_counts = np.zeros(shape=(n+1, k), dtype=int)
        for b in range(max_bin+1):
            _, y_counts[b, :] = np.unique(y[bins == b], return_counts=True)
        cond_entr = np.zeros(n+1)
        # cond_entr[:max_bin+1] = entropy(y_counts[:max_bin+1], axis=1, base=2)
        cond_entr[:max_bin+1] = [entropy_from_counts(each) for each in y_counts[:max_bin+1]]
        mean_cond_entr = sum(counts[:max_bin+1]*cond_entr[:max_bin+1])/n
        return Binning(x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr)

    def __init__(self, x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr):
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
        self.num_bins = len(np.unique(self.bins))-1 if -1 in self.bins else len(np.unique(self.bins))

    def move(self, i, dest):
        orig = self.bins[i]
        self.bins[i] = dest
        c = self.y[i]
        # print('before', self.counts[orig], self.counts[dest])
        self.mean_cond_entr = self.mean_cond_entr - self.counts[orig]*self.cond_entr[orig]/self.n - self.counts[dest]*self.cond_entr[dest]/self.n
        self.counts[orig] -= 1
        self.counts[dest] += 1
        if self.counts[orig] == 0:
            self.num_bins -= 1
        if self.counts[dest] == 1:
            self.num_bins += 1
        self.y_counts[orig, c] -= 1
        self.y_counts[dest, c] += 1
        self.cond_entr[orig] = incremental_entropy(self.cond_entr[orig], self.counts[orig]+1, self.y_counts[orig, c]+1, self.y_counts[orig, c])
        self.cond_entr[dest] = incremental_entropy(self.cond_entr[dest], self.counts[dest]-1, self.y_counts[dest, c]-1, self.y_counts[dest, c])
        self.mean_cond_entr = self.mean_cond_entr + self.counts[orig]*self.cond_entr[orig]/self.n + self.counts[dest]*self.cond_entr[dest]/self.n

    def best_cut_off(self, order,delta, typ):
        _max_bin = self.max_bin
        split_off_bins = np.ones(max(self.n, _max_bin)+1, dtype=int)*-1
        origins = np.zeros(self.n, dtype=int)
        k = len(np.unique(self.y))
        mean_cond_entr_star = self.mean_cond_entr # previous best mean_cond_entr
        num_pre_bins = self.num_bins
        bins, max_bin, counts, y_counts, cond_entr = deepcopy(self.bins), _max_bin, deepcopy(self.counts), deepcopy(self.y_counts), deepcopy(self.cond_entr)
        i_star = -1
        # forward
        for i in range(self.n):
            j = order[i] # real data index
            b = self.bins[j] # seek for real data bin index
            if split_off_bins[b] == -1:
                _max_bin += 1
                split_off_bins[b] = _max_bin

                if _max_bin == len(self.bins):
                    self.counts = np.append(self.counts, 0)
                    self.y_counts = np.vstack([self.y_counts,  np.zeros(shape=(1, k), dtype=int)])
                    self.cond_entr = np.append(self.cond_entr, 0)
                    self.bins = np.append(self.bins, -1)
                    split_off_bins = np.append(split_off_bins, -1)

            _b = split_off_bins[b]
            origins[i] = b # aovid creating empty bins
            self.move(j, _b)

            # delta = delta/(n*p-i)

            if mean_cond_entr_star > self.mean_cond_entr: 
                i_star, mean_cond_entr_star, num_pre_bins = order[i], self.mean_cond_entr, self.num_bins
                bins, max_bin, counts, y_counts, cond_entr = deepcopy(self.bins), _max_bin, deepcopy(self.counts),deepcopy(self.y_counts), deepcopy(self.cond_entr)
        # rewind
        for i in range(self.n-1, -1, -1):
            j = order[i]
            self.move(j, origins[i])
        
        return mean_cond_entr_star, i_star, bins, max_bin, counts, y_counts, cond_entr, num_pre_bins    

class efficientJointDiscretizationMI(Binning):

    def __init__(self, permut=False, duplicate=True, early_stopping=None, delta_correction=True, delta=0.05):
        self.permut = permut
        self.duplicate = duplicate
        self.early_stopping = early_stopping
        self.delta = delta
        self.delta_correction = delta_correction
    
    def fit(self, x, y, bins=None):
        pool = Pool()
        n, p = x.shape
        bins = np.zeros(n, dtype=int) if not bins else bins
        binning = super().from_assignment(x, y, bins)
        sigma = np.argsort(x, axis=0)
        best_mean_cond_entr_star = np.infty
        _, cnts = np.unique(y, return_counts=True)
        # entro_y = entropy(cnts/sum(cnts), base=2)
        entro_y = entropy_from_counts(cnts)
        values, step_fmi, dims_list = [], [], []
        orders = [sigma[:, j] for j in range(p)]
        col_nums = [_ for _ in range(p)]
        pre_mean_cond_entr_star = entro_y
        delta = self.delta
        num_pre_bins = 1
        while True:
            best_star = -1
            best_dim = -1
            result = pool.starmap(binning.best_cut_off, zip(orders, [self.delta for _ in orders], [self.early_stopping for _ in orders]))
            for j in range(len(result)):
                mean_cond_entr_star, i_star, bins, max_bin, counts, y_counts, cond_entr, num_after_bins = result[j]
                
                if best_mean_cond_entr_star > mean_cond_entr_star and i_star != -1:
                    best_dim = col_nums[j]
                    best_bins, best_max_bin, best_counts, best_y_counts, best_cond_entr, best_mean_cond_entr_star, best_star, best_num_after_bins = deepcopy(bins), max_bin, \
                                                                                                        deepcopy(counts), deepcopy(y_counts), \
                                                                                                        deepcopy(cond_entr), mean_cond_entr_star, i_star, num_after_bins
                
            degree_of_freedom = best_num_after_bins - num_pre_bins
            if self.delta_correction:
                if not self.duplicate:
                    delta = self.delta/(n*(p - len(values)-1))
                else:
                    delta = self.delta/(n*p - len(values)-1)
            if match_stopping(pre_mean_cond_entr_star, best_mean_cond_entr_star, size=n, degree_of_freedom=degree_of_freedom, delta=delta, typ=self.early_stopping):
                pre_mean_cond_entr_star = best_mean_cond_entr_star
                num_pre_bins = best_num_after_bins
                binning = Binning(x=x, 
                                y=y, 
                                bins=best_bins,
                                max_bin = best_max_bin,
                                counts = best_counts,
                                y_counts = best_y_counts,
                                cond_entr = best_cond_entr,
                                mean_cond_entr = best_mean_cond_entr_star)
                dims_list.append(best_dim)
                values.append(x[sigma[best_star, best_dim], best_dim])
                fmi = 1- best_mean_cond_entr_star/entro_y
                step_fmi.append(fmi)
                if not self.duplicate:
                    orders = [sigma[:, each] for each in col_nums if each != best_dim]
                    col_nums.remove(best_dim)
            else:
                pool.close()
                return dims_list, step_fmi, best_bins, values, best_counts, best_y_counts, best_cond_entr, len([each for each in best_counts if each])