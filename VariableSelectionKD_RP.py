from operator import ne
import numpy as np
from scipy.stats import norm, entropy, uniform, bernoulli
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.integrate import quad, dblquad
from multiprocessing import Pool, freeze_support
from itertools import combinations, repeat
import os

RNG = np.random.default_rng(seed = 0)

import pandas as pd
from MedianSplitTree import *
from estimators import *

var2 = 0.5
marginal_x1_pdf = uniform(-8, 16).pdf #norm(0, 4).pdf

def cond_mean_x2(x1):
    return x1+2*np.sin(10*x1/(2*np.pi))

def p_x2_given_x1(x2, x1):
    # notably re-implementing the normal density here seems to provide a factor 2 speed-up when integrating the joint pdf
    #     return norm.pdf(x2, loc=cond_mean_x2(x1), scale=var2**0.5)
    return np.exp(-(x2-cond_mean_x2(x1))**2/(2*var2))/(2*np.pi*var2)**0.5

def joint_x_pdf(x1, x2):
    return marginal_x1_pdf(x1)*p_x2_given_x1(x2, x1) #conditional_x2_pdf(x1)(x2)

# def joint_x_pdf(x1, x2):
#     return marginal_x1_pdf(x1)*np.exp(-(x2-cond_mean_x2(x1))**2/(2*var2))/(2*np.pi*var2)**0.5

def cond_y_probs(x1, x2):
    p = expit(-x1+x2-2)
    return np.array([1-p, p])

def cond_ent_y(x1, x2):
    return entropy(cond_y_probs(x1, x2))

def marginal_density_x2(x2):

    def integrand(x1):
        return joint_x_pdf(x1, x2)

    a, b = -8, 8
    p, _ = quad(integrand, a, b)
    return p

marginal_density_x2 = np.vectorize(marginal_density_x2)

def y_probs_given_x1(x1):
    
    def integrand(x2):
        p, q = cond_y_probs(x1, x2)*joint_x_pdf(x1, x2)
        return p

    p, _ = quad(integrand, -np.inf, np.inf)
    p = p / marginal_x1_pdf(x1)
    return np.array([1-p, p])

y_probs_given_x1 = np.vectorize(y_probs_given_x1, signature='()->(k)')

def y_probs_given_x2(x2):
    
    def integrand(x1):
        p, q = cond_y_probs(x1, x2)*joint_x_pdf(x1, x2)
        return p

    p, _ = quad(integrand, -8, 8)
    p = p / marginal_density_x2(x2)
    return np.array([1-p, p])

y_probs_given_x2 = np.vectorize(y_probs_given_x2, signature='()->(k)')

def weighted_cond_ent(x1, x2):
    return joint_x_pdf(x1, x2)*cond_ent_y(x1, x2)

def weighted_cond_prob_y(x1, x2):
    p, _ = cond_y_probs(x1, x2)
    return joint_x_pdf(x1, x2)*p

def weighted_cond_ent_y_given_x1(x1):
    return entropy(y_probs_given_x1(x1))*marginal_x1_pdf(x1)

def weighted_cond_ent_y_given_x2(x2):
    return entropy(y_probs_given_x2(x2))*marginal_density_x2(x2)

def rvs(n):
    x1 = uniform(-8, 16).rvs(n, RNG)
    x2 = norm.rvs(loc=cond_mean_x2(x1), scale=var2**0.5, random_state=RNG)
    y = bernoulli.rvs(expit(-x1+x2-2), random_state=RNG)
    return np.column_stack((x1, x2, y))

def generate_data(df, n, irr=100):
    for _ in range(3, irr):
        name = 'X' + str(_)
        stop = False
        num = uniform(-8, 16).rvs(n) 
        df[name] = num
    return df

def single_sel(predictor_target, types, tree_rep=1):
    """This function provides the minimum aic by given data
    Input:
            predictor_target: combination of tuples for example (data  y).
            types: tree type: 'kd' or 'rp'
            tree_rep: num of repetitions to select best aic
    Output:
            best_aic: global minimum aic value
            final_fmi: fmi based on best aic
    """
    data, y = predictor_target
    best_aic = np.infty
    for _ in range(tree_rep):
        aic, fmi = MedianSplitTree([data], y).makeWholeTree(types)
        if best_aic > aic:
            best_aic = aic
            final_fmi = fmi
    return best_aic, final_fmi

def variable_sel(predictors, y, chosen_lst, evaluation = 'aic_only', tree_types='rp', tree_rep=1, result_lst = [None, None], stop=False):
    """This is total algorithm for the variable selection. We first get the whole combination of predictors and then select one predictor which has minimum aic 
    values. After that, based on the previous selected predictors, we select a new predictor which can minimize the aic value.
    Input:
        predictors: predictors, can be a dataframe
        y: target variables, can be list/array
        chosen_lst: selected predictors name
        evaluation: criteria for example "aic-only" and 'averaging model' (not available now)
        tree_types: type of splitting, for example rp/kd refer to 'RP-tree/KD-tree'
        tree_rep: num of repetition to get local minimum aic value
        result_lst: store the aic, fmi
        stop: boolen operator, "False" if let this recursive function runs. 'True' to stop the function
    Output:
        chosen_lst: selected predictors name
        result_lst: best aic, fmi
    """
    if stop==True:
        return chosen_lst, result_lst

    potential_lst, data_lst = [], []
    for i in range(len(predictors.columns)):
        column = predictors.columns[i]
        if column in chosen_lst:
            continue
        data = tuple([tuple(each) for each in predictors[chosen_lst + [column]].values.tolist()])
        data_lst.append([data, y])
        potential_lst.append(column)
        
    freeze_support()
    pool = Pool()
    computation_lst = pool.starmap(single_sel, zip(data_lst, repeat(tree_types), repeat(tree_rep)))
    pool.close()

    prev_aic, prev_fmi = result_lst
    final_aic, final_fmi = result_lst
    variable = None

    # initial value
    if evaluation == 'aic_only':
        current_aic, current_fmi = computation_lst[0]
        column = potential_lst[0]
        for i in range(len(computation_lst)):
            if current_aic > computation_lst[i][0]:
                current_aic, current_fmi = computation_lst[i]
                column = potential_lst[i] # select best one
                
        if final_aic==None or final_aic > current_aic:
            final_aic, final_fmi, variable = current_aic, current_fmi, column

        if prev_aic == final_aic:
            stop=True
        if variable:
                chosen_lst.append(variable)
        result_lst = [final_aic, final_fmi]

    # For averaging model:
    # if evaluation == 'ave_model':
    #     for i in range(len(computation_lst)):
    #         fmi_dic, aic_dic = computation_lst[i][2:]
    #         aic_lst = list(aic_dic.values())
    #         fmi_lst = list(fmi_dic.values())
    #         # control weight
    #         weights = [math.exp(-each) for each in aic_lst]
    #         para = 1/sum(weights)
    #         weights = [para * each for each in weights]
    #         # comparison
    #         averaging_fmi = sum(np.array(weights) * np.array(fmi_lst))
    #         if final_fmi==None or final_fmi < averaging_fmi:
    #             variable = potential_lst[i]
    #             final_fmi = averaging_fmi

    #     if prev_fmi == final_fmi:
    #         stop=True
        
    #     if variable:
    #             chosen_lst.append(variable)
    #     result_lst = [final_aic, final_fmi]

    return variable_sel(predictors, y, chosen_lst, evaluation, tree_types, tree_rep, result_lst, stop)


def evaluate(tp, fp, tn, fn):
    recall = tp/(tp + fn)
    precision = tp/(tp+fp)
    try:
        f1 = recall*precision*2/(precision + recall)
    except:
        f1 = 0
    accuracy = (tp + tn)/(tp+fp+tn+fn)
    return accuracy, recall, precision, f1

import timeit
if __name__ == '__main__':

    rep=30 # control the reps
    for size in [20, 100, 500]: # may add more sample sizes
        start = timeit.default_timer()
        for typ in ['kd', 'rp']:
            for evaluation in ['aic_only']:
                aic_list = []
                fmi_list = []
                accuracy_list = []
                recall_list = []
                precision_list = []
                f1_list = []
                variable_lst = []
                dic = {}
                for _ in range(rep):
                    df = pd.DataFrame(rvs(size))
                    df.columns = ['X1', 'X2', 'Y']
                    df = generate_data(df, len(df), 100) # generate 100 irrelevant variables
                    y = df.Y.values.tolist()
                    predictors = df.drop(columns='Y')
                    variables, lst = variable_sel(predictors=predictors, y=y, chosen_lst=[], evaluation=evaluation, tree_types=typ, tree_rep=10) # vary the tree_repetitions
                    # evaluation
                    total_columns = list(predictors.columns)
                    pos_columns = ['X1', 'X2']
                    neg_columns = [each for each in total_columns if each not in pos_columns]
                    tp = sum([1 for each in variables if each in pos_columns])
                    fp = sum([1 for each in variables if each in neg_columns])
                    tn = len(neg_columns) - fp
                    fn = len(pos_columns) - tp
                    # print(evaluation, variables, tp, fp, fn, tn)
                    accuracy, recall, precision, f1 = evaluate(tp, fp, tn, fn)

                    variable_lst.append(variables)
                    aic_list.append(lst[0])
                    fmi_list.append(lst[1])
                    accuracy_list.append(accuracy)
                    recall_list.append(recall)
                    precision_list.append(precision)
                    f1_list.append(f1)
                name = str(size) + '_' + typ + '_' + evaluation
                dic_current = {'aic':aic_list, 'fmi': fmi_list, 'Acc': accuracy_list, 'Recall': recall_list, 'precision': precision_list, 'f1': f1_list, 'variables': variable_lst}
                directory = '/Users/yluu0028/Documents/Yiwen PhD/Confirmation/Year_2/VariableSelection/'
                df = pd.DataFrame(dic_current)
                df.to_csv(directory + name+'.csv', index=False)
        
        end = timeit.default_timer()
        print(size)
        print(end-start)
    
