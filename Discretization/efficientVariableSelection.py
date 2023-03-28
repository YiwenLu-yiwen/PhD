import numpy as np
from scipy.stats import chi2
from multiprocess import Pool
from copy import deepcopy
from incrementalBinning import Binning2

def cond_entr_obj(binning):
    return binning.mean_cond_entr

def update_alpha(org_alpha, n, p, i, criteria):
    """i means selected points in i-1 iteration
    """
    if criteria is None or criteria == 'bonferroni_duplicate' or criteria == 'sine_duplicate':
        return org_alpha/(n*p-i)
    elif criteria is None or criteria == 'orginal':
        return org_alpha
    elif criteria == 'bonferroni_unique' or criteria == 'sine_unique':
        return org_alpha/(n*(p-i))
    
def chi_square_obj(binning):
    """i means selected points in i-1 iteration
    """
    df = binning.non_empty_bin_count-binning.old_non_empty_bin_count
    return 1 - chi2.cdf(2*binning.n*(binning.old_mean_cond_entr-binning.mean_cond_entr), df*(binning.k_-1))


def chi_square_cond_entr_obj(binning):
    if binning.non_empty_bin_count==binning.old_non_empty_bin_count:
        return
    return (chi_square_obj(binning), cond_entr_obj(binning))

def create_cutpoint_index_obj(data_index_list, gamma):
    num_cutpoints = int(len(data_index_list)**gamma)
    base = int(100/num_cutpoints)
    cutpoint_range = [_*base for _ in range(num_cutpoints)]
    cutpoints_index = set()
    for each in cutpoint_range:
        cutpoints_index.add(int(np.ceil(np.percentile(data_index_list, each))))
    return sorted(list(cutpoints_index))

class VariableSelection:

    def __init__(self, alpha=0.05, base='mi', criteria='orginal', gamma=1, oracle=False):
        """
            base: We apply two different methods to compare cutpoints.
                mi: use pearson naive mutual information
                p_value: use chi square hypothesis test value
            criteria: We apply bonferroni correction in criteria: alpha/(np-i), where n is sample size, p is dimensions, 
                      i is the number of iteration.
                      We adapt sine correction, ranking p-value and ja/m, j is the rank of current cutpoint based on p-values,
                      m is the remain cutpoints.

                None: remove all criteria, will stop after selecting 100 cutpoints by using bonferroni_correction.
                bonferroni_duplicate: use bonferroni_correction but allow selecting duplicate variables.
                bonferroni_unique: use bonferroni_correction but allow selecting unique variables.
                sine_duplicate: use sine correction
                sine_unique: use sine correction
                orginal: keep orginal alpha
        """
        self.alpha = alpha
        self.base = base
        self.criteria = criteria
        self.gamma = gamma
        self.oracle = oracle
    
    def fit(self, x, y):
        binning = Binning2.trivial(x, y)
        binning.dim_p_value_ = []
        orders = np.argsort(x, axis=0)
        orders_new = np.argsort(x, axis=0)
        self.n_, self.p_ = x.shape
        self.num_cuts_selected_ = np.zeros(self.p_, int)
        dims_ = np.arange(self.p_)
        alpha = self.alpha
        t = 0
        pool=Pool()
        cutpoint_index = create_cutpoint_index_obj(np.arange(self.n_), self.gamma)
        if self.criteria in ['sine_duplicate', 'sine_unique'] and not self.oracle:
            obj = chi_square_cond_entr_obj
        elif self.base=='p_value':
            obj = chi_square_obj
        elif self.base=='mi':
            obj = cond_entr_obj
        n_ = len(cutpoint_index) # if multi-target, should be (k-1)*df
        while True:
            j_star, i_star, obj_star = -1, -1, float('inf')
            if n_ * self.p_ - t == 0 or self.p_ == t: break
            alpha = update_alpha(self.alpha, n_, self.p_, t, self.criteria)
            res = pool.starmap(binning.best_cut_off, zip([orders_new[:, _] for _ in range(orders_new.shape[1])], [obj for _ in range(orders_new.shape[1])], 
                                                        [cutpoint_index for _ in range(orders_new.shape[1])],
                                                        [self.criteria for _ in range(orders_new.shape[1])], 
                                                        dims_))

            if self.criteria in ['bonferroni_duplicate', 'bonferroni_unique', 'orginal'] or self.oracle:          
                for j in range(len(dims_)):
                    if res[j][1] < obj_star:
                        j_star, i_star, obj_star = dims_[j], res[j][0], res[j][1]
            elif self.criteria in ['sine_duplicate', 'sine_unique']:
                res = [code for each in res for code in each]
                res.sort()
                fail_cnt = 0
                for _rank in range(len(res)):
                    if res[_rank][0] > alpha * (_rank+1):
                        fail_cnt += 1
                        continue
                if (fail_cnt == _rank + 1) or not res:
                    break                    
                elif self.base == 'p_value':
                    j_star, i_star, obj_star = res[0][3], res[0][2], res[0][0]
                elif self.base == 'mi':
                    res = [(code[1], code[0], code[2]) for each in res for code in each]
                    res.sort()
                    j_star, i_star, obj_star = res[0][3], res[0][2], res[0][0]

            p_value = obj_star
            cond_ent_old = binning.mean_cond_entr
            params_old = binning.non_empty_bin_count
            binning.apply_cut_off(i_star, orders[:, j_star])
            cond_entr_new = binning.mean_cond_entr
            params_new = binning.non_empty_bin_count
            if self.criteria in ['bonferroni_duplicate', 'bonferroni_unique', 'orginal']:
                if self.base == 'mi':
                    p_value = 1 - chi2.cdf(2*self.n_*(cond_ent_old-cond_entr_new), (params_new-params_old)*(binning.k_-1))
                elif self.base == 'p_value':
                    p_value = obj_star
            
            if (self.criteria in ['bonferroni_duplicate', 'bonferroni_unique', 'orginal'] and p_value <= alpha) or self.criteria in ['sine_duplicate', 'sine_unique'] or self.oracle:
                self.num_cuts_selected_[j_star] += 1
                binning.old_mean_cond_entr = cond_entr_new
                binning.old_non_empty_bin_count = params_new
                binning.dim_p_value_.append((j_star, p_value))
                if self.criteria == 'bonferroni_unique' or self.criteria == 'sine_unique':
                    dims_ = [each for each in dims_ if each != j_star]
                    orders_new = orders[:, dims_]
            else:
                break
            if self.oracle and (len(binning.dim_p_value_) > 100 or cond_ent_old==cond_entr_new or params_new==params_old):
                break
            t += 1

        self.selected_ = np.flatnonzero(self.num_cuts_selected_)
        self.num_cuts_selected_ = self.num_cuts_selected_[self.selected_]
        return self, binning

    def transform(self, x, y):
        return x[:, self.selected_], y
    
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)