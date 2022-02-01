import numpy as np
import random
from estimators import *
import math
from scipy.spatial import distance


def get_node(a, dic, cnt, existing=None, orginal_tree=None):
    """creat leaf/node dictionary
    a: make tree values for example, tree.MakeTree(2)
    dic = {}, cnt = 0 for the initial settings
    """
    if a:
        if a.val not in dic:
            dic[a.val] = {"length":len(a.val), 'level':cnt, 'entropy': None, 'loglikelihood': None, 'main':None}
        if a.leftChild:
            cnt += 1
            dic[a.val].update({a.leftChild.val: {"length":len(a.leftChild.val), 'level':cnt, 'entropy': None, 'loglikelihood': None, 'main':None}})
        if a.rightChild:
            cnt += 1
            if a.leftChild:
                cnt -=1
            dic[a.val].update({a.rightChild.val:{"length":len(a.rightChild.val), 'level':cnt, 'entropy': None, 'loglikelihood': None, 'main':None}})
        
        dic = get_node(a.leftChild, dic, cnt)
        dic = get_node(a.rightChild, dic, cnt)
    dic = {key:dic[key] for key in list(dic.keys())[::-1]}
    return dic

def get_entropy(tree, y_dic, MinSize, estimator='naive_estimate'):
    """Get each node/leaf entropy using naive estimator
    x: x array
    y_dic: {xval:y}
    tree: make tree values for example, tree.MakeTree(2)
    """
    dic = get_node(a=tree, dic={}, cnt=0)
    depth = 0
    est = eval(estimator)
    dic_keys = list(dic.keys())
    dic_keys.sort(key=lambda s: len(s))
    
    for key in dic_keys:
        key_lst = list(dic[key].keys())
        for _ in ['length', 'level', 'entropy', 'loglikelihood', 'main']:
            key_lst.remove(_)
        if not key_lst:
            if not dic[key]['entropy']:
                data = np.array([y_dic[code] for code in key])
                dic[key]['entropy'] = est(data)
                dic[key]['loglikelihood'] = dic[key]['entropy'] * len(data)
                dic[key]['main'] = np.unique(data, return_counts=True)

        else:
            for each in key_lst:
                if not dic[each]['entropy']:
                    data = np.array([y_dic[code] for code in each])
                    dic[each]['entropy'] = est(data)
                    dic[each]['loglikelihood'] = dic[each]['entropy'] * len(data)
                    dic[key]['main'] = np.unique(data, return_counts=True)

            fir, sec = key_lst
            length = dic[fir]['length'] + dic[sec]['length'] 
            if dic[fir]['level'] != dic[sec]['level']:
                print('level', fir, sec)
            entro = dic[fir]['entropy'] * dic[fir]['length']/length + dic[sec]['entropy'] * dic[sec]['length']/length
            dic[key] = {'length': length, 'level': dic[fir]['level']-1, 'entropy': entro, 'loglikelihood': dic[fir]['loglikelihood'] + dic[sec]['loglikelihood']}

    depth = max([dic[each]['level'] for each in dic.keys()])
    num_leaves = sum([1 for each in dic.keys() if len(each) <= MinSize])
    return dic, num_leaves, depth

class RPTree():
    def __init__(self, val=None, rule=None, v=None):
        self.val = val
        self.rule = rule
        self.leftChild = None
        self.rightChild = None
        self.num = 0
        self.cnt = 0
        self.v = v

    def __str__(self):
        return str(self.val)

    def farest(self, data, x):
        """This is for RP-tree Max Version
        """
        init, point = -1, 0
        for each in data:
            try:
                temp = sum((np.array(each) - np.array(x))**2)
            except:
                temp = (np.array(each) - np.array(x))**2
            if temp > init:
                init = temp
                point = each
        return point, init

    def get_epilson(self, data, x):
        """This is for RP-tree Max Version
        """
        y, dist = self.farest(data, x)
        try:
            epilson = 6 * math.sqrt(dist)/math.sqrt(len(x))
        except:
            epilson = 6 * math.sqrt(dist)
        epilson = np.random.uniform(-1*epilson, epilson, 1)[0]
        return epilson

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

    def ChooseRule_kd(self, data):
        """This is for KD-Tree version
        """
        try:
            sel = random.randint(0,len(data[0])-1)
        except:
            sel = -1
        if sel == -1:
            rule = np.median(data)
            res_left = [each for each in data if each <= rule]
            res_right = [each for each in data if each not in res_left]
        else:
            samp = [_[sel] for _ in data]
            rule = np.median(samp)
            res_left = [each for each in data if each[sel] <= rule]
            res_right = [each for each in data if each not in res_left]
        if res_left and res_right:
            return rule, None, tuple(res_left), tuple(res_right), sel
        pass

    def ChooseRule_max(self, data):
        """This is RP-Tree MAX version
        """
        # choose a direction i
        stop=False
        while not stop:
            try:
                v = self.get_rand_vec(len(data[0]))
            except:
                v = self.get_rand_vec(1)
            proj_rules = [self.project(each, v) for each in data]
            rule = np.median(proj_rules)
            res_left = [each for each in data if self.project(each, v) <= rule]# This is RP-tree Max Version + self.get_epilson(data, each)] #
            res_right = [each for each in data if each not in res_left]
            if res_left and res_right:
                return rule, v, tuple(res_left), tuple(res_right)#, str_rule        

    def ChooseRule_median(self, data, c=1):
        """This is RP-Tree median version
        """
        try:
            v = self.get_rand_vec(len(data[0]))
        except:
            v = self.get_rand_vec(1)
        proj_rules = [self.project(each, v) for each in data]
        rule = np.median(proj_rules)
        dists = distance.cdist(data, data, 'euclidean')
        s = np.max(dists)
        sa = 0
        for each in data:
            y, dist = self.farest(data, each)
            sa += dist/len(data)**2
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
            return rule, v, tuple(res_left), tuple(res_right), None#, str_rule 

        else:
            try:
                rule = np.median([sum((each - np.mean(data, axis=0))**2) for each in data])
                res_left = [each for each in data if sum((each - np.mean(data, axis=0))**2) <= rule]
                res_right = [each for each in data if each not in res_left]
                choose = 1
            except:
                rule = np.median([(each - np.mean(data))**2 for each in data])
                res_left = [each for each in data if (each - np.mean(data))**2 <= rule]
                res_right = [each for each in data if each not in res_left]
                choose = 2
            return rule, None, tuple(res_left), tuple(res_right), choose #str_rule 

    def MakeTree(self, MinSize, existing=False, data=False, orginal_tree=None, types='median'):
        """Recursive function of RP Tree main process
        existing: if already have rules, return True. Otherwise, return False
        types: max/median/kd
        """
        if not existing and len(self.val) <= MinSize:
            return RPTree(self.val)
        if existing:
            if orginal_tree.leftChild==None:
                return RPTree(self.val)

        v, choose= None, None
        if not existing:
            if types == 'max':
                Rule, v, res_left, res_right = self.ChooseRule_max(self.val)
            elif types == 'median':
                Rule, v, res_left, res_right, choose = self.ChooseRule_median(self.val)
            elif types == 'kd':
                Rule, v, res_left, res_right, choose = self.ChooseRule_kd(self.val)
            self.rule, self.v = Rule, v
        else:
            Rule, v = orginal_tree.rule, orginal_tree.v
            self.val = data
            if types == 'max':
                res_left, res_right = self.comparison_max(Rule, v, data)
            elif types == 'median':
                res_left, res_right = self.comparison_median(Rule, v, data, choose)
            elif types == 'kd':
                Rule, v, res_left, res_right, choose = self.comparison_median(Rule, v, data, choose)
            

        leftree = RPTree(res_left, Rule, v)
        rightree = RPTree(res_right, Rule, v)
        if existing:
            LeftTree = leftree.MakeTree(MinSize, existing=existing, data=leftree.val, orginal_tree=orginal_tree.leftChild) 
            RightTree = rightree.MakeTree(MinSize, existing=existing, data=rightree.val, orginal_tree=orginal_tree.rightChild)
        else:
            LeftTree = leftree.MakeTree(MinSize, data=leftree.val, orginal_tree=None)
            RightTree = rightree.MakeTree(MinSize, data=rightree.val, orginal_tree=None) # existing=existing,
        self.leftChild = LeftTree
        self.rightChild = RightTree
        return self

    def comparison_max(self, rule, v, data):
        """This is for RP-Tree max comparison
        """
        res_left = [each for each in data if self.project(each, v) <= rule]# + self.get_epilson(data, each)]
        res_right = [each for each in data if each not in res_left] # self.project(each, v) > (rule + self.get_epilson(data, each))]
        return tuple(res_left), tuple(res_right)

    def comparison_median(self, rule, v, data, choose=None):
        """This is for RP-Tree median comparison
        """
        if choose == 1:
            res_left = [each for each in data if self.project(each, v) <= rule]
            res_right = [each for each in data if each not in res_left]
        else:
            try:
                res_left = [each for each in data if sum((each - np.mean(data, axis=0))**2) <= rule]
                res_right = [each for each in data if each not in res_left]
            except:
                res_left = [each for each in data if (each - np.mean(data))**2 <= rule]
                res_right = [each for each in data if each not in res_left]
        return tuple(res_left), tuple(res_right)

    def comparison_kd(self, rule, v, data, sel):
        """This is for KD-Tree comparison
        """
        res_left = [each for each in data if each[sel] <= rule]
        res_right = [each for each in data if each not in res_left]
        return tuple(res_left), tuple(res_right)
        

    def volumn(self, MinSize, data, orginal_tree):
        """This algorithm is for calculating volumn of data
        """
        return self.MakeTree(MinSize, True, data, orginal_tree)
    
    def counts(self):
        """Count leaves/bins
        """
        if not self:
            return 0
        if self.leftChild is None and self.rightChild is None:
            return 1
        else:
            return self.leftChild.counts() + self.rightChild.counts()
