import numpy as np
from multiprocess import Pool
from copy import deepcopy
import timeit
import pandas as pd
from data_generator import dataGenerator2d
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
    tp = sum([1 for each in selected_variables if each in pos_variables])
    fp = sum([1 for each in selected_variables if each in neg_variables])
    tn = len(neg_variables) - fp
    fn = len(pos_variables) - tp
    return evaluate(tp, fp, tn, fn)


class evaluationExperiment:

    def __init__(self, size, model_dic, dataGenerator=dataGenerator2d, rr=2, irr=48, verbose=True, data_rep=5, types='mixed_circle') -> None:
        self.size = size
        self.model_dic = model_dic
        self.verbose = verbose
        self.data_rep = data_rep
        self.dataGenerator = dataGenerator
        self.irr = irr
        self.rr = rr
        self.result = dict(zip(size, [{} for _ in size]))
        self.types = types
        self.dataframe = {}

    def run(self):
        pool = Pool()

        for size in self.size:
            if self.verbose:
                start = timeit.default_timer()
            for _ in range(self.data_rep):
                predictors, target = self.dataGenerator(n=size, irr=self.irr, types=self.types).fit()  # for train, only relevant
                total_columns = list(predictors.columns)
                pos_columns = ['X' + str(i) for i in range(self.rr)]
                neg_columns = [each for each in total_columns if each not in pos_columns]

                for model_name in self.model_dic:
                    tree = deepcopy(self.model_dic[model_name])
                    if model_name == 'linear':
                        best_subset, best_aic = tree(predictors, target)
                    elif model_name == 'joint' or model_name == 'stage':
                        data = deepcopy(predictors)
                        data['Y'] = target
                        best_subsetData_list, best_aic, dim_list, best_value_list = tree(data, permut=True)
                        best_subset = np.unique(['X' + str(each) for each in dim_list])
                    else:
                        best_subset, best_aic = variable_sel(predictors, target, pool, tree, best_subset=[])
                    if model_name not in ['linear', 'mi']:
                        best_aic = abs(best_aic)

                    accuracy, recall, precision, f1 = evaluation_result(best_subset, pos_columns, neg_columns)
                    if model_name not in self.result[size]:
                        self.result[size][model_name] = {'scores': [best_aic], 'variables': [best_subset], 'accuracy': [accuracy],
                                                   'recall': [recall], 'precision': [precision], 'f1': [f1]}
                    else:
                        self.result[size][model_name]['variables'].append(best_subset)
                        self.result[size][model_name]['scores'].append(best_aic)
                        self.result[size][model_name]['accuracy'].append(accuracy)
                        self.result[size][model_name]['recall'].append(recall)
                        self.result[size][model_name]['precision'].append(precision)
                        self.result[size][model_name]['f1'].append(f1)
            if self.verbose:
                end = timeit.default_timer()
                print('Sample size:', size)
                print('Time(s):', end - start)
        pool.close()
        return self.result

    def summary(self):
        for each in self.result:
            df = pd.DataFrame(self.result[each]).T
            df['scores'] = df['scores'].apply(lambda x: np.mean(x))
            df['accuracy'] = df['accuracy'].apply(lambda x: np.mean(x))
            df['precision'] = df['precision'].apply(lambda x: np.mean(x))
            df['recall'] = df['recall'].apply(lambda x: np.mean(x))
            df['f1'] = df['f1'].apply(lambda x: np.mean(x))
            self.dataframe[each] = df
        return self.dataframe