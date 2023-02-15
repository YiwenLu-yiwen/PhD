import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.stats import multinomial 
import pandas as pd
from math import sin, log, exp, sqrt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class _power:
    """Power transformation
    """    
    def fit(self, variable):
        return variable ** 3

    def __str__(self) -> str:
        return "power"

class _square:
    """Square transformation
    """    
    def fit(self, variable):
        return variable ** 2

    def __str__(self) -> str:
        return "square"

class _square_root:
    """Square root transformation
    """    
    def fit(self, variable):
        return abs(variable) ** 0.5
    
    def __str__(self) -> str:
        return "square Root"

class _log:
    """Log transformation
    """    
    def fit(self, variable):
        return np.array(list(map(log, abs(variable))))
    
    def __str__(self) -> str:
        return "log"

class _sin:
    """Sine transformation
    """  
    def fit(self, variable):
            return np.array(list(map(sin, variable)))
    
    def __str__(self) -> str:
        return "sin"

class _exp:
    """Exponential transformation
    """     
    def fit(self, variable):
        return np.array(list(map(exp, variable)))

    def __str__(self) -> str:
        return "exp"

class _None:
    """No transformation, return itself
    """
    def fit(self, variable):
        return variable

    def __str__(self) -> str:
        return "no transformation"

class transformation:
    """Transformation class
    Contain different transformation functions
    fit functions are used to randomly choose one transformation method
    """

    def __init__(self) -> None:
        self.func = None

    def randchoose(self):
        functions = np.array([_sin(), _power(), _square(), _exp(), _square_root(), _log(), _None()])
        index = [np.random.choice(len(functions), 1)][0]
        return functions[index][0]
    
    def fit(self, data):
        func = self.randchoose()
        return func, func.fit(data)


class structureDesignBasic:
    def __init__(self, org_p, total_p, inter_order=1, alpha=0.5) -> None:
        """Only consider first order
        : param
            org_p: number of orginal feature space, int
            total_p: number of total output features
            inter_order: interaction order (e.g. inter_order=1, means "X_iX_j",
                                            e.g. inter_order=2, means "X_iX_jX_k"
                                            )
            alpha: control the weight of main effect variables and interaction variables 
        """
        self.p=org_p
        self.alpha=alpha
        self.maxorder=inter_order
        self.total_p=total_p

    
    def generate_individual(self):
        return np.array(['X' + str(i) for i in range(self.p)])

    def define_combination(self, maxorder):
        lst = self.generate_individual()
        result = []
        for i in range(len(lst)):
            for j in combinations(lst, maxorder):
                if len(j) == maxorder and i == 0:
                    result.append(j)
                elif len(j) == maxorder and j[0] > lst[i-1]:
                    result.append(j)
        return result

    def generate_interaction(self, maxorder):
        result = []
        for i in range(2, maxorder+2):
            result += self.define_combination(i)
        return np.array(result)

    def select_var(self, alpha, length, var_array, prop=None):
        prop = int(alpha * self.total_p) if not prop else prop
        prop = length if length < prop else prop
        return var_array[np.random.choice(length, prop, replace=False)].tolist(), prop

    def sample(self):
        individual = self.generate_individual()
        interaction = self.generate_interaction(self.maxorder)
        individual, prop = self.select_var(alpha=self.alpha, 
                                    length=len(individual),
                                    var_array=individual,
                                    prop=None)
        individual = [[each] for each in individual]
        interaction, prop = self.select_var(alpha=self.alpha, 
                                    length=len(interaction),
                                    var_array=interaction,
                                    prop=self.total_p-prop)
        assert len(individual) + len(interaction) == self.total_p, "Not match total number of predictors"
        return individual, interaction

class structureDesign(structureDesignBasic):
    """This is only structure design, in which, it only contans columns names and rules.
    This class outputs number of relevant number, relevant variable list, predictors, 
    interaction terms and single terms. Note, we only connsider unique relevant variables. 
    For example, if "X1", "X1X2" both exist, there will be considered as 2 variables.
    """
    
    def __init__(self, org_p, r, total_p, inter_order=2, alpha=0.5, interaction=False) -> None:
        super().__init__(org_p, total_p, inter_order, alpha)
        self.r = r
        self.interaction = interaction

    def simplefit(self):
        """Only record one time
        """
        individual, interaction = super().sample()
        if self.interaction:
            p = individual + interaction
            index = np.random.permutation(len(p))
            p = np.array(p)[index].tolist()
        else:
            p = individual
            index = np.random.permutation(len(p))
            p = np.array(p)[index].tolist() + interaction
        r_list, inter_terms, single_terms, i = [], [], [], 0
        r = len(r_list)
        while r < self.r:
            select_variable = p[i]
            if len(select_variable) == 1:
                if select_variable not in r_list:
                    r_list.append(select_variable)
                    single_terms.append(select_variable)
            elif len(select_variable) > 1:
                for each in select_variable:
                    if each not in r_list:
                        r_list.append(each)
                inter_terms.append(select_variable)
            r = len(r_list)
            i += 1
        return r, r_list, p, inter_terms, single_terms
    
    def sample(self):
        """match relevant variables
        """
        stop=False
        while not stop:
            r, r_list, p, inter_terms, single_terms = self.simplefit()
            if r == self.r:
                if (self.interaction and inter_terms) or (not self.interaction and not inter_terms):
                    return r, r_list, p, inter_terms, single_terms


class dataTransform(structureDesign):
    """data transformation class
    """
    
    def __init__(self, org_p, r, total_p, inter_order=2, alpha=0.5, interaction=True, transf=None):
        super().__init__(org_p, r, total_p, inter_order, alpha, interaction)
        self.transf=transf

    def sample(self):
        r, r_list, p, inter_terms, single_terms = super().sample()
        if self.transf:
            for i in range(len(p)):
                p[i] = list(p[i])
                p[i].append(transformation().randchoose())
        else:
            for i in range(len(p)):
                p[i] = list(p[i])
                p[i].append(_None())
        r_list = [each[0] for each in r_list]
        inter_terms = [each[:-1] for each in inter_terms]
        single_terms = [each[0] for each in single_terms]
        return r, r_list, p, inter_terms, single_terms

class generateDataBasic(dataTransform):
    """Combine data transformation and X generator
    This function generate X, beta, and use optimized function to optimize SNR by adjusting beta
    In addition, this algorithm control the input variance to be 1 (g(X)/sqrt(var(g(X)))) and use beta_0 = E(g(X)), 
    where g(X) is the transformation of X
    """

    def __init__(self, n, rho, org_p, r, total_p, inter_order=2, alpha=0.5, SNR=1, interaction=False, transf=None, model='binomial') -> None:
        self.org_p = org_p
        super().__init__(org_p, r, total_p, inter_order, alpha, interaction, transf)
        self.n=n
        self.rho=rho
        self.SNR=SNR
        self.model=model

    def generateX(self):
        org_p = self.org_p
        # Generate the design matrix
        rho_matrix = np.zeros((org_p, org_p))
        for i in range(org_p):
            for j in range(org_p):
                rho_matrix[i,j] = self.rho ** abs(i-j)
        X = np.random.multivariate_normal(np.zeros(org_p), rho_matrix, self.n)
        df_X = pd.DataFrame(dict(zip(['X' + str(i) for i in range(org_p)], X.T)))
        return df_X

    def testX(self):
        r, r_list, p, inter_terms, single_terms = super().sample()
        df_X = self.generateX()
        test_X, col_name = [], []
        for variables in p:
            x_hat = 1
            name = ''
            for _var in variables[:-1]:
                x_hat *= df_X[_var].values
                name += _var
            X_i = variables[-1].fit(x_hat)
            X_i = X_i/sqrt(np.var(X_i)) # normalized
            test_X.append(X_i)
            col_name.append(name)
        b0 = -np.sum(np.mean(test_X))
        return r, r_list, p, np.array(test_X), col_name, b0
        
    def generateBtrue(self, p):
        # Generate 'btrue'
        Innz = np.arange(self.r)
        btrue = np.zeros(p)
        btrue[Innz] = np.random.standard_t(10, self.r)
        return btrue
    
    def glmpredict(self, X, btrue, b0, model):
        n, p = X.shape
        eta = np.dot(X, btrue) + b0
        if model == 'binomial': ## currently only working on "binomial"
            mu = 1/(1+ np.exp(-eta))
            v = mu * (1 - mu)
            y = np.array([each[0] for each in np.random.rand(n, 1)]) < mu
            y = np.array(list(map(int, y)))
            snr = np.sqrt(np.mean(mu**2)/np.mean(v))

        elif model == 'multinomial':
            mu = []
            for i in range(p):
                if btrue[i]:
                    eta_hat = np.dot(X[:, i], btrue[i]) + b0
                    mu.append(np.mean(np.exp(eta_hat)))
            mu = mu/sum(mu)
            v = [p*(1-p) for p in mu]
            y = multinomial.rvs(1, mu, n)
            snr = np.sqrt(np.mean(mu**2)/np.mean(v))

        return snr, y, mu, v
    
    def optimize_func(self, x, X, btrue, b0, model, SNR):
        """Optimize function
        """
        return ((self.glmpredict(X, np.exp(x)*btrue, b0, model)[0] - SNR)**2).mean()

    def sample(self):
        stop=False
        while not stop:
            r, r_list, p, X, col_name, b0 = self.testX()
            btrue = self.generateBtrue(len(col_name))
            if self.model not in ['binomial', 'multinomial']:
                raise Exception("Only accept binomial and multinomial")
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
            stop=True
        snr, y, mu, v = self.glmpredict(X.T, np.exp(g.x)*btrue, b0, self.model)
        if self.model=='multinomial':
            y = [np.where(each == 1)[0][0] for each in y]
        elif self.model == 'binomial':
            y=y
        df_X = pd.DataFrame(dict(zip(col_name, X)))
        return df_X, y, r_list, snr

class generateData(generateDataBasic):

    def __init__(self, n, rho, org_p, r, total_p, inter_order=2, alpha=0.5, SNR=1, interaction=False, transf=None, model='binomial') -> None:
        super().__init__(n, rho, org_p, r, total_p, inter_order, alpha, SNR, interaction, transf, model)
    
    def sample(self):
        stop=False
        while not stop:
            df_X, y, r, snr = super().sample()
            if len(np.unique(y)) > 1 and abs(snr-self.SNR) < 0.001:
                print(snr)
                stop=True
                return df_X, y, r