import numpy as np
from multiprocess import Pool
from copy import deepcopy
import timeit
import pandas as pd
from binning import equal_width
from joint_lognalgorithmPvalue import jointDiscretizationPvalue#, stageWiseDiscretizationP
# from joint_n2algorithm1 import simpleJointDiscretization1
from joint_nalgorithmPvalue import efficientJointDiscretizationPvalue
from joint_nalgorithmMI import efficientJointDiscretizationMI
from baseline_models import random_forest
from efficientVariableSelection import VariableSelection

import warnings
warnings.filterwarnings("ignore")

def single_sel(parameters, tree):
    # find best rp_tree , pairs, subset, targes 
    """data is the sublist of predictors in X, y_dic is the values of Y
    parameters: combination of tuples for example (data  y).
    """
    data, y = parameters
    all_result, model = tree.fit(data, y)
    best_score, k, all_rules = all_result
    return best_score 

def variable_sel(predictors, y, pool, tree, best_subset=[]):
    """This is total algorithm for the variable selection
    """
    remain_columns = [each for each in predictors.columns if each not in best_subset]
    best_score, prev_score, stop = -np.infty, None, False
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
        
        score_lst = pool.starmap(single_sel, zip(data_lst, tree_lst))

        # initial value
        current_best_score = score_lst[0]
        column = potential_lst[0]
        for i in range(len(score_lst)):
            if current_best_score < score_lst[i]:
                current_best_score = score_lst[i]
                column = potential_lst[i] # select best one
        
        if best_score < current_best_score:
            best_score, variable = current_best_score, column

        if prev_score == best_score:
            stop=True
        else:
            prev_score = best_score
            
        if variable and variable not in best_subset:
            best_subset.append(variable)
            
    return best_subset, best_score

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
    selected_variables = list(set(selected_variables))
    tp = sum([1 for each in selected_variables if each in pos_variables])
    fp = sum([1 for each in selected_variables if each in neg_variables])
    tn = len(neg_variables) - fp
    fn = len(pos_variables) - tp
    return evaluate(tp, fp, tn, fn)


class evaluationExperiment:

    def __init__(self, data, target, model_dic, pos_variables, neg_variables, verbose=True, oracle=False) -> None:
        self.model_dic = model_dic
        self.verbose = verbose
        self.result = {}
        self.data = data
        self.target = target
        self.pos_columns = pos_variables
        self.neg_columns = neg_variables
        self.oracle = oracle

    def run(self):
        pool = Pool()
        if self.verbose:
            start = timeit.default_timer()

        predictors, target = self.data, self.target
        pos_columns = self.pos_columns
        neg_columns = self.neg_columns

        for model_name in self.model_dic:
            # add time here
            model_starttime = timeit.default_timer()
            tree = deepcopy(self.model_dic[model_name])
            if type(tree) in [VariableSelection]:
                result = tree.fit(deepcopy(predictors).values, target)
                dim_list, best_aic, _bins, n_vars  = tree.selected_, [], None, None
                best_subset = ['X' + str(each) for each in dim_list]
            elif type(tree) in [efficientJointDiscretizationMI, efficientJointDiscretizationPvalue]:
                result = tree.fit(deepcopy(predictors).values, target) 
                dim_list, best_aic, _bins, n_vars  = result[0], result[1], result[-1], result[3]   
                best_subset = ['X' + str(each) for each in dim_list]
            else:
                best_aic, best_subset = tree.fit(predictors, target)
                _bins, n_vars = -1, -1
                
            if model_name not in ['linear', 'mi']:
                try:
                    best_aic = abs(best_aic)
                except:
                    best_aic = best_aic
            
            if type(tree) in [random_forest] or self.oracle: # random forest oracle settings
                recall, precision, f1 = 0, 0, 0
                _variables = []
                for each in best_subset:
                    _variables.append(each)
                    accuracy, recall_hat, precision_hat, f1_hat = evaluation_result(_variables, pos_columns, neg_columns)
                    if f1_hat > f1:
                        recall, precision, f1 = recall_hat, precision_hat, f1_hat
                        best_variables = deepcopy(_variables)
                if f1 == 0:
                    best_subset = deepcopy(best_subset)
                else:
                    best_subset = deepcopy(best_variables)
            else:
                accuracy, recall, precision, f1 = evaluation_result(best_subset, pos_columns, neg_columns)
            model_endtime = timeit.default_timer()
            model_time = model_endtime - model_starttime
            self.result[model_name] = {'scores': [best_aic], 'variables': best_subset, 'accuracy': [accuracy],
                                        'recall': [recall], 'precision': [precision], 'f1': [f1], 'time':[model_time],
                                        'n_bins': [_bins], 'n_vars': [n_vars]}
        if self.verbose:
            end = timeit.default_timer()
            print('Time(s):', end - start)
        pool.close()
        return self.result

    def summary(self):
        for each in self.result:
            df = pd.DataFrame(self.result[each]).T
            self.dataframe[each] = df
        return self.dataframe