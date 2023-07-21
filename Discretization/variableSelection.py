import numpy as np
from copy import deepcopy
import timeit
import pandas as pd
from efficientVariableSelection import VariableSelectionOracle, VariableSelectionReliable
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

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

def _AUC_ROC_(rr, pred_rr, p=100):
    """AUC_ROC
    """
    all_vars = np.zeros(p)
    all_vars[rr] = 1
    _rank = np.argsort(pred_rr)
    return 1-roc_auc_score(all_vars, _rank)

class evaluationExperiment:

    def __init__(self, data, target, model_dic, rr=None, verbose=True) -> None:
        self.model_dic = model_dic
        self.verbose = verbose
        self.result = {}
        self.data = data
        self.target = target
        self.rr = rr

    def run(self):
        if self.verbose:
            start = timeit.default_timer()
        predictors, target = self.data, self.target
        _n, _p = self.data.shape
        for model_name in self.model_dic:
            print('begin', model_name)
            model_starttime = timeit.default_timer()
            model = deepcopy(self.model_dic[model_name])
            if type(model) in [VariableSelectionOracle, VariableSelectionReliable]:
                fitted_model, fitted_binning = model.fit(deepcopy(predictors), target, rr_vars=self.rr)
                mi_list = [each[-2] for each in fitted_binning.dim_point_value_]
                model_endtime = timeit.default_timer()
                model_time = model_endtime - model_starttime
                self.result[model_name] = {'increased_mi': mi_list, 
                                        'score_feature_rank': list(fitted_model.score_feature_rank()),
                                        'order_feature_rank': fitted_model.order_feature_rank(),
                                        'time':[model_time], 
                                        'n_bins': [fitted_binning.non_empty_bin_count],
                                        'back_up_info': fitted_binning.dim_point_value_,
                                        'score_AUC_ROC': _AUC_ROC_(self.rr, list(fitted_model.score_feature_rank()), p=_p),
                                        'order_AUC_ROC': _AUC_ROC_(self.rr, fitted_model.order_feature_rank(), p=_p),
                                        'rr_vars': list(self.rr)}
            else:
                model_endtime = timeit.default_timer()
                model.fit(deepcopy(predictors), target)
                model_time = model_endtime - model_starttime
                self.result[model_name] = {'increased_mi': [None], 
                                        'score_feature_rank': model.score_feature_rank(),
                                        'order_feature_rank': model.score_feature_rank(),
                                        'time':[model_time], 
                                        'n_bins': [None],
                                        'back_up_info': [None],
                                        'score_AUC_ROC': _AUC_ROC_(self.rr, model.score_feature_rank(), p=_p),
                                        'order_AUC_ROC': _AUC_ROC_(self.rr, model.score_feature_rank(), p=_p),
                                        'rr_vars': list(self.rr)}
            print('end', model_name)
        if self.verbose:
            end = timeit.default_timer()
            print('Time(s):', end - start)

    def summary(self):
        return pd.DataFrame(self.result).T