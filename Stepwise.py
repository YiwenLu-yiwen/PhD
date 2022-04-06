import numpy as np
from parallelCuttingPlan import *
from itertools import combinations
from permutation import *
from binning import *
from functions import *
from estimators import * 
            
def reliable_MI1d(x, y):
    """
    calculate reliable fraction of mi (discrete variables)
    """
    return (gain(np.array(x), np.array(y)) - Permutation(np.array(x), np.array(y)).summary()) / entropy(np.array(y))

def estimate_MI1d(x, y):
    """
    calculate fraction of mi
    """
    return fGain(np.array(x), np.array(y))
            
def convert_range(x, cutPointlst):
    """
    This algorithm is to convert cutPointList to specific range
    x: matrix
    """
    if cutPointlst == []:
        return [(min(x), max(x))]
    else:
        return [(min(x)[0,0]-1, cutPointlst[0])] + [(cutPointlst[i], cutPointlst[i+1]) for i in range(len(cutPointlst)-1)] + [(cutPointlst[-1], max(x)[0,0] +1)]
    
def Stepwise(df, m=0, current=None, candi_fmi=0, stop_indx=None):
    """We use james-shrinkage estimator
    """
    data = df[:, m]
    y= df[:, -1]
    stop = False
    known_cut_lst, candi_current,candi_known_cut = [], [],[]
    cut_points = getCandiPoint(data, equal_freq, 2*int(math.log2(len(data))))
    # candi_known_cut = cut_points
    while not stop:
        for k in cut_points:
            if k in known_cut_lst:
                continue
            if known_cut_lst == cut_points:
                stop=True
                print('may need more points')
                break
            curr_bins = cutPointBin(data, known_cut_lst, k)
            if current:
                combine_bins = [tuple([current[i], curr_bins[i]]) for i in range(len(curr_bins))]
                tags = np.unique(combine_bins, axis = 0)
                tags = [tuple(each) for each in tags]
                tags_dict = dict(zip(tags, [_ for _ in range(len(tags))]))
                combine_bins = [tags_dict[each] for each in combine_bins]
            else:
                combine_bins = curr_bins
            comb = np.array([(combine_bins[i], y[i]) for i in range(len(y))])
            curr_fmi = (james_estimate(np.array(combine_bins)) + james_estimate(np.array(y)) - james_estimate(comb))/james_estimate(np.array(y))
            if candi_fmi < curr_fmi:
                candi_fmi = curr_fmi
                candi_known_cut = [k]
                candi_current = combine_bins
        # get result
        known_cut_lst += candi_known_cut.copy()
        if not candi_known_cut:
            stop=True
            break
        candi_known_cut = []
        # print(known_cut_lst, candi_current, candi_fmi)
    m += 1
    if m > df.shape[1] - 2 or m == stop_indx:
        bins_num = len(np.unique(combine_bins))
        return candi_fmi, bins_num
    return Stepwise(df, m, candi_current, candi_fmi, stop_indx)
