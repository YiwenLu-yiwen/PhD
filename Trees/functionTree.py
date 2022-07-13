import math
import numpy as np
import random   
from estimators import naive_estimate
from copy import deepcopy
from sklearn.utils import check_X_y

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def log_loss(groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        likelihood = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p != 0:
                    score += -p * math.log2(p) * size # for likelihood
            # weight the group score by its relative size
            likelihood += score * (size / n_instances)
        return likelihood


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = np.infty, np.infty, np.infty, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            loss = log_loss(groups, class_values)
            if loss < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], loss, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

class FunctionTree:
    """This is only for random projection median splitting tree. We want to use this algorithm handling continuous variables.
    We create a branch of trees and select best AIC tree. However, we find that the minimum AIC values always appear in the first split.
    With number of split increases, AIC increases and loglikelihood decreases. May not work in the variable selection.
    """
    
    def __init__(self, option='kd', AIC=True, estimator='naive_estimate'):
        """Choose to use different options and different AIC/BIC criteria
        """
        self.probs = None
        self.all_rules = [[]]
        self.dict = {}
        self.aic = np.infty
        self.AIC = AIC
        self.fmi = 0
        self.option = option
        self.val = None
        self.y_entropy = None
        self.target = None
        self.y_dic = None
        self.length = None
        self.num_leaves = None

    def __str__(self):
        return str(self.val)

    def get_rand_vec(self, dims):
        """
        Get random direction i 
        """
        components = [np.random.normal() for i in range(dims)]
        r = math.sqrt(sum(x*x for x in components))
        v = [x/r for x in components]
        return np.array(v)

    def project(self, x, y):
        """
        Get the projection of x on y
        """
        return np.dot(x, y) / np.linalg.norm(y)
    
    def nodeLoglikelihood(self, leaf_lst: list, n: int, AIC: bool) -> tuple:
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
        def prob(leaf):
            return sum(leaf)/len(leaf) 
        temp_all = []
        y_leaves = []
        for leaf in leaf_lst:
            temp = []
            for code in leaf:
                temp.append(self.y_dic[tuple(code)])
            temp_all.append([leaf, temp])
            y_leaves.append(temp)
        leafLikelihood = np.array([naive_estimate(np.array(code))*len(code) for code in y_leaves]) # each H(Y|X=x) * c(x, y)
        probs = [prob(each) for each in y_leaves]
        # p(y|x')H(y|x=x')
        weight = np.array([len(code[0]) for code in leaf_lst])
        weight_sum = sum(weight)
        k = len(weight) 
        prob = weight/weight_sum
        # H(Y|X)
        tree_loglikelihood = sum(leafLikelihood)
        # fraction of MI
        fmi = 1 - tree_loglikelihood/self.y_entropy if self.y_entropy else 0 
        # AICc values
        if self.AIC:
            aic = 2* tree_loglikelihood + 2*n*k/(n-k-1)
        else:
            aic = 2* tree_loglikelihood + k*math.log(n) #+ 2*n*k/(n-k-1)  # apply bic here # if n !=k+1 and n!=k else 2*tree_loglikelihood + 2*k*n # 2* k/2 * math.log2(n) 
        return aic, fmi, tree_loglikelihood, k, probs

    
    def ChooseRule_kd(self, data, y=None):
        """This is for KD-Tree version
        """
        try:
            sel = random.randrange(0,len(data[0]))
            # print(len(data[0]))
            samp = [_[sel] for _ in data]
            rule = np.median(samp)
            res_left = [each for each in data if each[sel] <= rule]
            res_right = [each for each in data if each not in res_left]
        except:
            sel = -1
            rule = np.median(data)
            res_left = [each for each in data if each <= rule]
            res_right = [each for each in data if each not in res_left]
        # if res_left and res_right:
        return res_left, res_right, (sel,rule)
    
    def ChooseRule_median(self, data, y=None): # change it to mean accidentaly
        """This is RP-Tree median version. Currently, we just use random projection median split rather than using any distance function. 
        """       
        try:
            v = self.get_rand_vec(len(data[0]))
        except:
            v = self.get_rand_vec(1)
        proj_rules = [self.project(each, v) for each in data]
        rule = np.median(proj_rules)
        # rule = np.mean(proj_rules) 
        res_left = [each for each in data if self.project(each, v) < rule]
        res_right = [each for each in data if each not in res_left]
        return res_left, res_right, (v,rule)

    def ChooseRule_class(self, data, y=None):
        """Select one dimension, use all the data
        """
        y = [self.y_dic[tuple(each)] for each in data]
        dataset = []
        for i in range(len(data)):
            dataset.append(np.append(np.array(data[i]), y[i]))
        dataset = np.array(dataset)
        dic = get_split(dataset)
        sel, rule = dic['index'], dic['value']
        res_left =  [each[:-1] for each in dic['groups'][0]]
        res_right =  [each[:-1] for each in dic['groups'][1]]
        return res_left, res_right, (sel,rule)
        

    def fit(self: object, predictor:list, target:list):
        """After each splitting, we use AIC criteria to control how to split the nodes. The output is the global minimum AIC values and its FMI
        Input: option: 'kd', 'rp', 'classification'
        Output: aic: best tree aic values
                fmi: not fixed, ignore now
                k: number of parameters:
                rules: rule details
                self: best tree
        """
        self.val = [tuple([tuple(each) for each in predictor.values.tolist()])]
        self.y_entropy = naive_estimate(np.array(target))
        self.target = target
        self.y_dic = dict(zip(self.val[0], target))
        self.length = len(target)
        self.num_leaves = int(math.log2(len(target)))

        length = self.length
        stop, non_cnts, pre_aic=False, 0, None
        cnt = 0
        split_lst = []
        rules_lst = []
        while not stop:
            # print(len(self.all_rules), len(self.val), cnt, )
            # print(len(self.val[0]), len(self.val[0][0]))
            copy_data = deepcopy(self.val)
            split_lst = []
            copy_rules = deepcopy(self.all_rules)
            rules_lst = []
            cnt += 1
            while copy_data:
                # for one iteration split
                node = copy_data.pop()
                del_rules = copy_rules.pop()
                if self.option == 'kd':
                    tree_left, tree_right, rule = self.ChooseRule_kd(node)
                elif self.option == 'rp':
                    tree_left, tree_right, rule = self.ChooseRule_median(node)
                elif self.option == 'classification':
                    tree_left, tree_right, rule = self.ChooseRule_class(node)

                if tree_left:
                    temp_rules = deepcopy(del_rules)
                    temp_rules.append((self.option, rule[0], rule[1], '<'))
                    rules_lst += [temp_rules]
                    split_lst += [tree_left]
                if tree_right:
                    temp_rules = deepcopy(del_rules)
                    temp_rules.append((self.option, rule[0], rule[1], '>'))
                    rules_lst += [temp_rules]
                    split_lst += [tree_right]
                
                # if not tree_left or not tree_right:
                #     print(len(temp_rules), len(temp_lst))
                #     continue

                temp_rules = rules_lst + copy_rules
                temp_lst = split_lst + copy_data # if copy_data else split_lst
                aic, fmi, logs, k, probs = self.nodeLoglikelihood(temp_lst, length, self.AIC)
                if pre_aic == None or aic < pre_aic:
                    non_cnts = 0
                    result = [aic, fmi, k, temp_rules, probs]
                    pre_aic=aic
                else:
                    non_cnts += 1
                if non_cnts > 50: #apply early stopping, if 50 more leaves can't improve the tree, we don't generate anymore
                    stop=True
                    print(len(temp_rules))
                    break

            self.all_rules = deepcopy(temp_rules)
            self.val = deepcopy(temp_lst)
            # print(len(self.all_rules), len(self.val), stop, 'end')

            if length < 100:
                if len(self.val) >= length/2-1: #>= 20 or len(self.val) >= length/2:  #== length:
                    stop=True
            elif len(self.val) > length/3-2:
                stop=True
            if non_cnts > 50:
                stop=True
        self.all_rules, self.probs = result[-2:]

        return result[:-1], self

    def predict(self, data, y):
        """Return risk and log loss for given data, usually training sample gives values.
        Return:
            partition_lst: data distribution list
            loss: log loss
            risk: average error rate
            ids_lst: each data locates in which leaf
        """
        data = [tuple(each) for each in data.values.tolist()]
        y_dic = dict(zip(data, y))
        location_lst = [[] for _ in range(len(self.probs))]
        partition_lst = deepcopy(location_lst)
        ids_dic = {}
        def check_single_rule(point, rule):
            option, dim, bound, symbol = rule
            if option =='kd' or option == 'classification':
                if symbol == '>':
                    return point[dim] > bound
                else:
                    return point[dim] <= bound
            if option =='rp':
                if symbol == '>':
                    return self.project(point, dim) > bound
                else:
                    return self.project(point, dim) <= bound
        for each in data:
            for i in range(len(self.all_rules)):
                current_check = [check_single_rule(each, rule) for rule in self.all_rules[i]]
                if np.array(current_check).all():
                    location_lst[i].append(y_dic[each])
                    ids_dic[tuple(each)] = i
                    partition_lst[i].append(each)

        def logloss(lst, p):
            loss = 0
            for each in lst:
                if p == 1:
                    if each == 1:
                        loss += -math.log2(p)*each #-math.log2(p)*each -math.log2(1-p)*(1-each)]
                    else:
                        loss += np.infty
                elif p == 0:
                    if each == 1:
                        loss += np.infty #-math.log2(p)*each -math.log2(1-p)*(1-each)]
                    else:
                        loss += -math.log2(1-p)*(1-each)
                else:
                    loss += -math.log2(p)*each -math.log2(1-p)*(1-each)
            return loss
        loss = 0
        for i in range(len(location_lst)):
            loss += logloss(location_lst[i], self.probs[i])/len(y)

        def error(lst, p):
            p = 1 if p >= 0.5 else 0
            return sum([each != p for each in lst])
        risk = 0
        for i in range(len(location_lst)):
            risk += error(location_lst[i], self.probs[i])/len(y)

        ids_lst = ids_dic.values()
        return partition_lst, loss, risk, ids_lst