import numpy as np
from scipy.stats import chi2
from multiprocess import Pool
from copy import deepcopy
from incrementalBinning import Binning2

def cond_entr_obj(binning):
    return binning.mean_cond_entr

def update_delta(org_delta, n, p, i, criteria):
    """i means selected points in i-1 iteration
    """
    if criteria is None or criteria == 'holm_duplicate':
        return org_delta/(n*p-i)
    elif criteria is None or criteria == 'orginal':
        return org_delta
    elif criteria == 'holm_unique':
        return org_delta/(n*(p-i))
    
def chi_square_obj(binning):
    """i means selected points in i-1 iteration
    """
    df = binning.non_empty_bin_count-binning.old_non_empty_bin_count
    return 1 - chi2.cdf(2*binning.n*(binning.old_mean_cond_entr-binning.mean_cond_entr), df)

def create_cutpoint_index_obj(data_index_list, num_cutpoints):
    base = int(100/num_cutpoints)
    cutpoint_range = [_*base for _ in range(num_cutpoints)]
    cutpoints_index = set()
    for each in cutpoint_range:
        cutpoints_index.add(int(np.ceil(np.percentile(data_index_list, each))))
    return sorted(list(cutpoints_index))

class VariableSelection:

    def __init__(self, delta=0.05, base='mi', criteria='orginal', num_cutpoints=None):
        """
            base: We apply two different methods to compare cutpoints.

                mi: use pearson naive mutual information
                p_value: use chi square hypothesis test value

            criteria: We apply holm correction in criteria: delta/(np-i), where n is sample size, p is dimensions, 
                      i is the number of iteration.

                None: remove all criteria, will stop after selecting 100 cutpoints by using holm_correction.
                holm_duplicate: use holm_correction but allow selecting duplicate variables.
                holm_unique: use holm_correction but allow selecting unique variables.
                orginal: keep orginal delta
        """
        self.delta = delta
        self.base = base
        self.criteria = criteria
        self.num_cutpoints = num_cutpoints
    
    def fit(self, x, y):
        binning = Binning2.trivial(x, y)
        k = len(np.unique(y))
        orders = np.argsort(x, axis=0)
        self.n_, self.p_ = x.shape
        dims_ = np.arange(self.p_)
        selected = np.zeros(self.p_, bool)
        delta = self.delta
        t = 0
        pool=Pool()
        selected_cuts_ = []
        selected_mean_cond_entr_ = []
        if self.num_cutpoints is not None:
            cutpoint_index = create_cutpoint_index_obj(np.arange(self.n_), self.num_cutpoints)
        else:
            cutpoint_index = None
        while True:
            j_star, i_star, obj_star = -1, -1, float('inf')
            obj = cond_entr_obj if self.base == 'mi' else chi_square_obj
            res = pool.starmap(binning.best_cut_off, zip([orders[:, _] for _ in range(orders.shape[1])], [obj for _ in range(orders.shape[1])], 
                                                        [cutpoint_index for _ in range(orders.shape[1])]))
            for j in range(len(dims_)):
                if res[j][1] < obj_star:
                    j_star, i_star, obj_star = j, res[j][0], res[j][1]
            cond_ent_old = binning.mean_cond_entr
            params_old = binning.non_empty_bin_count
            binning.apply_cut_off(i_star, orders[:, j_star])
            cond_entr_new = binning.mean_cond_entr
            params_new = binning.non_empty_bin_count
            if self.base == 'mi':
                p_value = 1 - chi2.cdf(2*self.n_*(cond_ent_old-cond_entr_new), (params_new-params_old)*(k-1))
            elif self.base == 'p_value':
                p_value = obj_star
            n_ = self.n_ if self.num_cutpoints is None else self.num_cutpoints # if multi-target, should be (k-1)*df
            delta = update_delta(self.delta, n_, self.p_, t, self.criteria)
            if p_value <= delta:
                selected[j_star] = True
                selected_cuts_.append((j_star, x[orders[i_star, j_star], j_star]))
                selected_mean_cond_entr_.append((j_star, cond_entr_new))
            else:
                break
            t += 1
            binning.old_mean_cond_entr = cond_entr_new
            binning.old_non_empty_bin_count = params_new
            if self.criteria == 'holm_unique':
                dims_ = np.where(dims_ != j_star)[0]
                orders = orders[:, dims_]

        self.selected_ = np.flatnonzero(selected)
        return self, binning, selected_cuts_, selected_mean_cond_entr_

    def transform(self, x, y):
        return x[:, self.selected_], y
    
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)