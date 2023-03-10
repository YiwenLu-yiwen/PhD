import numpy as np
import pandas as pd

from math import sqrt
from scipy.special import comb
from copy import deepcopy, copy
from scipy.optimize import minimize
from scipy.stats import multinomial 
from IPython.display import Markdown

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def truncated_multinomial_rv(n, probs, bounds, random_state=None):
    RNG = np.random.default_rng(random_state)
    while True:
        counts = multinomial.rvs(n, probs, random_state=RNG)
        if (counts <= bounds).sum()==len(counts):
            return counts


def enum_comb_w_replacement(n, k):
    """Show how many combinations
    n: number of variables
    k: a specific degree
    """
    def rightmost_change_incr_index(a):
        n = len(a)
        for i in range(n-2, -1, -1):
            if a[i] > 0:
                return i+1

    def rest_from_index(a, i):
        s = sum(a[i:])
        a[i] = s
        for j in range(i+1, len(a)): a[j]=0

    def next(a):
        res = copy(a)
        i = rightmost_change_incr_index(res)
        res[i-1] = res[i-1]-1
        res[i] = res[i]+1
        rest_from_index(res, i)
        return res

    current = [k]
    current.extend((n-1)*[0])
    res = [current]

    last = (n-1)*[0]
    last.append(k)
    while current!=last:
        current = next(current)
        res.append(current)
    return res

def is_pure(monomial):
    non_zero = 0 
    for i in range(len(monomial)):
        non_zero += (monomial[i] > 0)
    return non_zero == 1

def sample_polynomial(n_vars, num_terms, degree, alpha=None, beta=0.5, random_state=None):
    RNG = np.random.default_rng(random_state)
    alpha = np.ones(degree)/degree if alpha is None else alpha
    while True:
        b = comb(n_vars, np.arange(1, degree+1), repetition=True)
        c = truncated_multinomial_rv(num_terms, alpha, b, random_state=RNG)
        # print(c)
        res = set()
        for i in range(degree):
            order_i = set()
            # if c[i] <= b[i]/2:
            while len(order_i) < c[i]:
                if RNG.random()<=beta:
                    if beta==1 and i==0: # fix the bugs if we only need interaction terms
                        break
                    monomial = tuple(multinomial.rvs(i+1, np.ones(n_vars)/n_vars, random_state=RNG))
                    if not is_pure(monomial): order_i.add(monomial)
                else:
                    var_idx = RNG.choice(n_vars)
                    monomial = [0]*n_vars
                    monomial[var_idx] = i+1
                    order_i.add(tuple(monomial))
            res = res.union(order_i)
        res = np.array(list(res))
        cnt=0
        for i in range(res.shape[1]):
            if len(np.unique(res[:, i])) != 1:
                cnt += 1
        if cnt == res.shape[1]:
            return res

class dataGenerator:

    def __init__(self, n, p, r, num_terms, degree, alpha=None, beta=0.5, random_state=None, rho=0.8, SNR=1, model='binomial') -> None:
        self.n=n
        self.p=p
        self.r=r
        self.num_terms=num_terms
        self.degree=degree
        self.alpha=alpha
        self.beta=beta
        self.random_state=random_state
        self.rho=rho
        self.SNR=SNR
        self.model=model

    def generateX(self):
        """Generate original feature space X
        """
        # Generate the design matrix
        rho_matrix = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                rho_matrix[i,j] = self.rho ** abs(i-j)
        X = np.random.RandomState(seed=self.random_state).multivariate_normal(np.zeros(self.p), rho_matrix, self.n)
        I = np.random.RandomState(seed=self.random_state).permutation(self.p)
        X = X[:, I]# permut
        df_X = pd.DataFrame(dict(zip(['X' + str(i) for i in range(self.p)], X.T)))
        return df_X

    def generateBtrue(self, k):
        """Generate 'btrue'
        """
        btrue = np.random.normal(0, 1, k)
        return btrue

    def testX(self):
        """Generate normalized feature space X'
        """
        variable_lst = sample_polynomial(self.r, self.num_terms, self.degree, 
                                        self.alpha, self.beta, random_state=self.random_state)
        df_X = self.generateX()
        test_X = []
        # generate b_i
        btrue = self.generateBtrue(self.num_terms)
        formula = []
        for j in range(len(variable_lst)):
            variable = variable_lst[j]
            X_i = np.zeros(self.n)
            X_i[:] = 1
            term = ""
            for i, order in enumerate(variable):
                X_i *= deepcopy(df_X.iloc[:, i].values) ** order
                if order:
                    if order ==1:
                        term+='X_{' + str(i) + '}'
                    else:
                        term+='X_{' + str(i) + "}^" + str(order)
            formula.append(term)
            btrue[j] = btrue[j]/sqrt(np.var(X_i)) # normalized
            test_X.append(X_i)
        return df_X, np.array(test_X), btrue, variable_lst, '+'.join(formula)

    def glmpredict(self, X, btrue, b0, model):
        n, p = X.shape
        eta = np.dot(X, btrue) + b0
        if model == 'binomial': ## currently only working on "binomial"
            mu = 1/(1+ np.exp(-eta))
            v = mu * (1 - mu)
            y = np.array([each[0] for each in np.random.rand(n, 1)]) < mu
            y = np.array(list(map(int, y)))
            snr = np.sqrt(np.var(mu)/np.mean(v))
        return snr, y, mu, v
    
    def optimize_func(self, x, X, btrue, b0, model, SNR):
        """Optimize function
        """
        return ((self.glmpredict(X, np.exp(x)*btrue, b0, model)[0] - SNR)**2).mean()

    def sample(self):
        while True:
            df_X, X, btrue, variable_lst, formula = self.testX()
            b0=0
            if self.model not in ['binomial', 'multinomial']:
                raise Exception("Only accept binomial and multinomial")
            if self.random_state:
                np.random.RandomState(seed=self.random_state)
            try:
                g = minimize(self.optimize_func, x0=0, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            btrue = np.exp(g.x)*btrue
            snr, y, mu, v = self.glmpredict(X.T, btrue, b0, self.model)
            if len(set(y)) != 1:
                break
        # print(snr)
        return df_X, y, variable_lst, btrue, Markdown("$$" + formula + '$$')
