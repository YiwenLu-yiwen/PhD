import math
import numpy as np
import random   
from estimators import *
    
class MedianSplitTree():
    """This is random projection splitting tree. We want to use this algorithm handling continuous variables.
    For each leave splitting, we use minimum AIC criteria.
    """
    
    def __init__(self, value_lst, target, estimator='naive_estimate'):
        self.val = value_lst
        self.dict = {}
        self.aic = np.infty
        self.fmi = 0
        self.est = eval(estimator)
        self.y_entropy = self.est(np.array(target))
        self.target = target
        self.y_dic = dict(zip(value_lst[0], target))
        self.num_leaves = math.log(len(value_lst[0]))

    def __str__(self):
        return str(self.val)
    
#     def farest(self, data, x):
#         """This is for RP-tree Max Version currently not use this function
#         """
#         init, point = -1, 0
#         for each in data:
#             try:
#                 temp = sum((np.array(each) - np.array(x))**2)
#             except:
#                 temp = (np.array(each) - np.array(x))**2
#             if temp > init:
#                 init = temp
#                 point = each
#         return point, init
    
    def get_rand_vec(self, dims):
        """
        Get random direction i 
        """
        x = np.random.standard_normal(dims)
        r = np.sqrt((x*x).sum())
        return x / r

    def project(self, x, y):
        """
        Get the projection of x on y
        """
        return np.dot(x, y) / np.linalg.norm(y)
    
    def nodeLoglikelihood(self, leaf_lst: list) -> tuple:
        """Get each node entropy/loglikelihood using an MI_estimator
        Input:
            leaf_lst: a built RP_tree last layer leaves list
            y_dic: a dictionary contains for example if assuming k targets {value1: target1, value2:target2,.....,value_k:target_k}
            MinSize: maximum size in each leaf
            estimator: mutual information estimator, default is naive estimator
        Output:
            nodedic: implement the tree_dic by only consider the leaves
            num_leaves: number of leaves 
        """
        temp_all = []
        for leaf in leaf_lst:
            temp = []
            for code in leaf:
                temp.append(self.y_dic[tuple(code)])
            temp_all.append(temp)
        leafLikelihood = np.array([self.est(np.array(code)) for code in temp_all])
        weight = np.array([len(code) for code in leaf_lst])
        weight_sum = sum(weight)
        prob = weight/weight_sum
        mi = sum(leafLikelihood * prob)
        fmi = 1- mi/self.y_entropy if self.y_entropy else 0
        tree_loglikelihood = [-math.log(each) if each != 0 else 0 for each in leafLikelihood]
        aic = 2*(len(weight) + sum(tree_loglikelihood))
        return aic, fmi
    
    def ChooseRule_kd(self, data):
        """This is for KD-Tree version
        """
        try:
            sel = random.randint(0,len(data[0])-1)
            samp = [_[sel] for _ in data]
            rule = np.median(samp)
            res_left = [each for each in data if each[sel] <= rule]
            res_right = [each for each in data if each not in res_left]
        except:
            sel = -1
            rule = np.median(data)
            res_left = [each for each in data if each <= rule]
            res_right = [each for each in data if each not in res_left]
        if res_left and res_right:
            return res_left, res_right
        
    def ChooseRule_median(self, data, c=10):
        """This is RP-Tree median version. Currently, we just use random projection median split rather than using any distance function. 
        """
        try:
            v = self.get_rand_vec(len(data[0]))
        except:
            v = self.get_rand_vec(1)
        
        # proj_rules = [self.project(each, v) for each in data]
        # rule = np.median(proj_rules)
        # dists = distance.cdist(data, data, 'euclidean')
        # s = np.max(dists)
        # sa = 0
        # for each in data:
        #     y, dist = self.farest(data, each)
        #     sa += dist/len(data)**2
        sa = 100
        s = 10
        # rp-median
        if c * sa >= s**2:            
            try:
                v = self.get_rand_vec(len(data[0]))
            except:
                v = self.get_rand_vec(1)
            proj_rules = [self.project(each, v) for each in data]
            rule = np.median(proj_rules)
            res_left = [each for each in data if self.project(each, v) <= rule]
            res_right = [each for each in data if each not in res_left]
            return res_left, res_right

        else:
            try:
                rule = np.median([sum((each - np.mean(data, axis=0))**2) for each in data])
                res_left = [each for each in data if sum((each - np.mean(data, axis=0))**2) <= rule]
                res_right = [each for each in data if each not in res_left]
            except:
                rule = np.median([(each - np.mean(data))**2 for each in data])
                res_left = [each for each in data if (each - np.mean(data))**2 <= rule]
                res_right = [each for each in data if each not in res_left]
            return res_left, res_right
        

    def makeWholeTree(self: object, option:str):
        """After each splitting, we use AIC criteria to control how to split the nodes. The output is the global minimum AIC values and its FMI
        Input: option: 'kd' or 'rp'
        Output: self.aic: minimum aic value for the whole tree
                self.fmi: fmi based on the minimum aic value
        """
        candidate_data, candidate_fmi, default_aic = [], self.fmi, self.aic
        stop = False
        counter = 0
        while not stop:
            for indx, node in enumerate(self.val):
                if len(node) == 1:
                    continue
                copy_data = self.val.copy()
                copy_data.pop(indx)

                if option == 'kd':
                    tree_left, tree_right = self.ChooseRule_kd(node) 
                elif option == 'rp':
                    tree_left, tree_right = self.ChooseRule_median(node)

                # split_lst = [tree_left] if tree_left else []
                split_lst = [tree_left] + [tree_right] #if tree_right else split_lst
                split_lst = split_lst + copy_data if copy_data else split_lst 
                            
                aic, fmi = self.nodeLoglikelihood(split_lst)
                if aic < default_aic:
                    default_aic = aic
                    candidate_fmi = fmi
                    candidate_data = split_lst.copy()

            if len(self.val) == self.num_leaves:
                stop = True
                continue

            if default_aic == self.aic:
                stop=True
                continue

            self.dict[counter] = [default_aic, candidate_fmi, candidate_data]
            self.val = candidate_data.copy()
            self.aic, self.fmi = default_aic, candidate_fmi
            counter += 1
        return self.aic, self.fmi
