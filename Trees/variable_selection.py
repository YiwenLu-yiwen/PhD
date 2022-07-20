from pickle import TRUE
import numpy as np
import pandas as pd
from functionTree import rpTree, kdTree, classifcationTree, honestTree
# from baseline_models import forward_selection
from multiprocess import Pool #, freeze_support
from itertools import combinations, repeat
from copy import deepcopy
from scipy.stats import norm, entropy, uniform, bernoulli
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.integrate import quad, tplquad
from sklearn.model_selection import KFold
import os

PROJECT_ROOT_DIR = "./Trees"
OUTPUTPATH = os.path.join(PROJECT_ROOT_DIR, "output")

RNG = np.random.default_rng(seed = 0)

var2 = 0.5
marginal_x1_pdf = uniform(-8, 8).pdf # norm(0, 4).pdf  

def cond_mean_x2(x1):
    return x1+2*np.sin(10*x1/(2*np.pi))


def p_x2_given_x1(x2, x1):
    return np.exp(-(x2-cond_mean_x2(x1))**2/(2*var2))/(2*np.pi*var2)**0.5


def cond_mean_x3(x1, x2):
    return x1 * x2 #+ 2*np.sin(10*x1*x2/(2*np.pi))


def p_x3_given_x1x2(x1, x2, x3):
    return np.exp(-(x3-x1*cond_mean_x2(x1))**2/(2*var2))/(2*np.pi*var2)**0.5


def joint_x_pdf(x1, x2, x3):
    return marginal_x1_pdf(x1)*p_x2_given_x1(x2, x1) * p_x3_given_x1x2(x1, x2, x3)


def cond_y_probs(x1, x2, x3):
    p = expit(-x1+x2+x3-2)
    return np.array([1-p, p])


def cond_ent_y(x1, x2, x3):
    return entropy(cond_y_probs(x1, x2, x3))


def weighted_cond_ent(x1, x2, x3):
    return joint_x_pdf(x1, x2, x3)*cond_ent_y(x1, x2, x3)

def weighted_cond_prob_y(x1, x2, x3):
    p, _ = cond_y_probs(x1, x2, x3)
    return joint_x_pdf(x1, x2, x3)*p

def rvs(n, irr=100):
    x1 = norm(0, 1).rvs(size=n)
    x2 = norm(0, 1).rvs(size=n)
    x3 = norm(0, 1).rvs(size=n)

    y = bernoulli.rvs(expit(x1+x2+x3), random_state=RNG)
    
    irr_lst = [uniform(-2, 2).rvs(n) for _ in range(irr)]
    for each in [x1, x2, x3]:
        irr_lst.append(each)
    irr_columns = ['X' + str(i) for i in range(4, irr+4)]
    rr_columns = ['X1', 'X2', 'X3']
    cols = irr_columns + rr_columns
    df = pd.DataFrame(np.column_stack(irr_lst))
    df.columns = cols
    return df, y

def single_sel(parameters, tree):
    # find best rp_tree , pairs, subset, targes 
    """data is the sublist of predictors in X, y_dic is the values of Y
    parameters: combination of tuples for example (data  y).
    """
    data, y = parameters
    best_aic = np.infty
    all_result, model = tree.fit(data, y)
    best_aic, final_fmi, k, all_rules = all_result
    return best_aic 

def variable_sel(predictors, y, pool, tree, best_subset=[]):
    """This is total algorithm for the variable selection
    """
    remain_columns = [each for each in predictors.columns if each not in best_subset]
    best_aic, prev_aic, stop = np.infty, None, False
    while not stop:
        potential_lst, data_lst, tree_lst, variable = [], [], [], None
        for i in range(len(remain_columns)):
            column = remain_columns[i]
            if column in best_subset:
                continue
            data = tuple([tuple(each) for each in predictors[best_subset + [column]].values.tolist()])
            data_lst.append([data, y])
            tree_lst.append(deepcopy(tree))
            potential_lst.append(column)
        
        aic_lst = pool.starmap(single_sel, zip(data_lst, tree_lst))

        # initial value
        current_best_aic = aic_lst[0]
        column = potential_lst[0]
        for i in range(len(aic_lst)):
            if current_best_aic > aic_lst[i]:
                current_best_aic = aic_lst[i]
                column = potential_lst[i] # select best one
        
        if best_aic > current_best_aic:
            best_aic, variable = current_best_aic, column

        if prev_aic == best_aic:
            stop=True
        else:
            prev_aic = best_aic
            
        if variable and variable not in best_subset:
            best_subset.append(variable)
            
    return best_subset, best_aic

def evaluate(tp, fp, tn, fn):
    recall = tp/(tp + fn)
    try:
        precision = tp/(tp+fp)
    except:
        precision = 0
    try:
        f1 = recall*precision*2/(precision + recall)
    except:
        f1 = 0
    accuracy = (tp + tn)/(tp+fp+tn+fn)
    return accuracy, recall, precision, f1


def evaluation_result(selected_variables, pos_variables, neg_variables):
    """Combine the evaluation part to get the accuracy, recall, precision, f1
    """
    tp = sum([1 for each in selected_variables if each in pos_variables])
    fp = sum([1 for each in selected_variables if each in neg_variables])
    tn = len(neg_variables) - fp
    fn = len(pos_variables) - tp
    return evaluate(tp, fp, tn, fn)


if __name__ == '__main__':
    import pandas as pd
    import timeit
    import warnings
    warnings.filterwarnings("ignore")
    # from DataGenerator import *
    variable_lst = []
    aic_list = []
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    types_list = []
    pool = Pool()
    model_dic = {'rp': rpTree(),
            'kd': kdTree(),
            'classification': classifcationTree(),
            'honest': honestTree()}

    for size in [20, 50, 100, 500, 1000, 2000, 4000, 8000]: # may add more sample sizes
        start = timeit.default_timer()
        for _ in range(5):
            predictors, target = rvs(size, 50)  # for train, only relevant
            
            total_columns = list(predictors.columns)
            pos_columns = ['X1', 'X2', 'X3']
            neg_columns = [each for each in total_columns if each not in pos_columns]

            for model_name in model_dic:
                tree = deepcopy(model_dic[model_name])
                best_subset, best_aic = variable_sel(predictors, target, pool, tree, best_subset=[])
                accuracy, recall, precision, f1 = evaluation_result(best_subset, pos_columns, neg_columns)
                variable_lst.append(best_subset)
                aic_list.append(best_aic)
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                precision_list.append(precision)
                f1_list.append(f1)
                types_list.append(model_name)

        file_name = str(size) + '.csv'
        dic_current = {'aic': aic_list, 'Acc': accuracy_list, 'Recall': recall_list, 
                        'precision': precision_list, 'f1': f1_list, 'variables': variable_lst, 'types': types_list}
        df = pd.DataFrame(dic_current)
        # df.to_csv(OUTPUTPATH + '/' + file_name, index=False)
        df.to_csv(os.path.join(OUTPUTPATH, file_name), index=False)
        end = timeit.default_timer()
        print(size)
        print(end-start)
    pool.close()
