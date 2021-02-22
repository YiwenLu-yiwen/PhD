from sklearn.base import BaseEstimator
from pygam.terms import TermList, te, SplineTerm
from pygam import LinearGAM, LogisticGAM
from old_functions import *


class VariableStructureGAM:

    def __init__(self, indi=None, multi=None, kind=LogisticGAM):
        self.kind = kind
        self.GAM_ = None
        self.multi_variables = multi
        self.indi_variables = indi

    def gam(self, lam=10, max_iter=1000):
        terms = TermList()
        if self.multi_variables:
            if self.indi_variables:
                for each in self.multi_variables :
                    if each[0] in self.indi_variables and each[1] in self.indi_variables:
                        self.indi_variables.remove(each[0])
                        self.indi_variables.remove(each[1])
                    terms += te(each[0], each[1], lam=lam)
        if self.indi_variables:
            for each in self.indi_variables:
                terms += SplineTerm(each, lam=lam)
        self.GAM_ = self.kind(terms, max_iter=max_iter)
        return self.GAM_

    def fit(self, x, y, lam=10, max_iter=1000):
        self.GAM_ = self.gam(lam=lam, max_iter=max_iter).fit(x, y)
        return self

    def predict(self, x):
        return self.GAM_.predict(x)

    def predict_proba(self, x):
        pass

    def summary(self):
        return self.GAM_, self.multi_variables, self.indi_variables