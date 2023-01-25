import numpy as np
from scipy.stats import norm, bernoulli
from scipy.special import expit
import pandas as pd
import sklearn.datasets as dt
from sklearn import cluster
from math import sin

class dataGenerator2d:
    """
    This algorithm is only for simple experiments
    """
    def __init__(self, n, irr, rr, types='independent_circle', factor=0.5, noise=0.05) -> None:
        self.n = n
        self.irr = irr
        self.types = types
        self.factor = factor
        self.noise = noise
        self.rr = rr
    
    def fit(self):
        if self.types == 'make_friedman1':
            x, y = dt.make_friedman1(self.n, self.irr + 5)
            y = bernoulli.rvs(abs(np.cos(y)))
            df = pd.DataFrame(x)
            colnames = ['X' + str(_) for _ in df.columns]
            df.columns = colnames
            return df, y

        elif self.types == 'independent_circle':
            X, y = dt.make_circles(n_samples=self.n, factor=self.factor, noise=self.noise)
            x1,x2 = X[:, 0], X[:, 1]

        elif self.types == 'mixed_circle':
            X, y = dt.make_circles(n_samples=self.n, factor=self.factor, noise=self.noise)
            x1,x2 = X[:, 0], X[:, 1]
            params = {'n_neighbors': 10, 'n_clusters': 2}
            average = cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="average"
        )
            y = average.fit_predict(X)

        elif self.types == 'no_structure' or self.types == 'irrelevant_only':
            x1 = norm(0, 1).rvs(size=self.n)
            x2 = norm(0, 1).rvs(size=self.n)
            y = bernoulli.rvs(expit(x1-x2))

        elif self.types == 'interaction':
            x1 = norm(0, 1).rvs(size=self.n)
            x2 = norm(0, 1).rvs(size=self.n)
            x3 = x1 * x2
            y = bernoulli.rvs(expit(x1-x2+x3))
        
        irr_lst = [norm(0, 1).rvs(size=self.n) for _ in range(self.irr)]
        if self.types == 'interaction':
            for each in [x3, x2, x1]:
                irr_lst.insert(0, each)
            irr_columns = ['X' + str(i) for i in range(3, self.irr+3)]
            rr_columns = ['X0', 'X1', 'X2']
            cols = rr_columns + irr_columns
            df = pd.DataFrame(np.column_stack(irr_lst))
            df.columns = cols
            return df, y
        else:
            rr_columns = []
            if self.types != 'irrelevant_only':
                for each in [x2, x1]:
                    irr_lst.insert(0, each)
                rr_columns = ['X0', 'X1']
            irr_columns = ['X' + str(i) for i in range(2, self.irr+2)]
            cols = rr_columns + irr_columns
            df = pd.DataFrame(np.column_stack(irr_lst))
            df.columns = cols
            return df, y
        
# all independent variables
class dataGenerator:
    """
    Linear
    Non-linear
    Real_data
    """
    """Aim to generate n variables with some partial relationships
    """

    def __init__(self, n, irr, rr, types='linear') -> None:
        self.n = n
        self.irr = irr
        self.types = types
        # self.factor = factor # maybe use these weights in the future
        # self.noise = noise
        self.rr = rr

    def generate_independent(self, num_variables):
        predictor_list = [norm(0, 1).rvs(size=self.n) for _ in range(num_variables)]
        return predictor_list

    def generate_bernoulli_target(self, predictor_list):
        return bernoulli.rvs(expit(sum(predictor_list)))


    ########## relationship ##########
    def generate_power(self, variable):
        return np.array(variable) ** 3

    def generate_square(self, variable):
        return np.array(variable) ** 2
    
    def generate_square_root(self, variable):
        return np.array(variable) ** 0.5   

    def generate_sin(self, variable):
        return np.array(list(map(sin, variable)))
    
    def generate_2_power(self, variable):
        return 2 ** variable
    ##################################
    def fit(self):
        if self.types == 'linear':
            predictor_list, target =  self.fit_linear()
        elif self.types == 'non_linear':
            predictor_list, target =  self.fit_non_linear()
        irr_list = self.generate_independent(self.irr)
        predictor_list += irr_list
        rr_columns = ['X' + str(i) for i in range(self.rr)]
        irr_columns = ['X' + str(i) for i in range(self.rr, self.irr+self.rr)]
        cols = rr_columns + irr_columns
        df = pd.DataFrame(np.column_stack(predictor_list))
        df.columns = cols
        return df, target

    def fit_linear(self):
        predictor_list = self.generate_independent(self.rr)
        target = self.generate_bernoulli_target(predictor_list)
        return predictor_list, target

    def fit_non_linear(self):
        # Y = \sum x_i^3 + x_^2 + x^0.5 
        # ^3
        default_variables = self.generate_independent(self.rr)
        # ^2

        # \sqrt

        pass



if __name__=='__main__':
    rr = 2 # relevant variables number
    irrelevant_number = 0 # irrelevant variables number
    sample_size = 100
    generate_type = 'independent_circle'
    """
    independent_circle: 2 relevant variables
    mixed_circle: 2 relevant variables
    no_structure: 2 relevant variables
    interaction: 3 relevant variables
    """
    dataExp, yExp = dataGenerator2d(rr=rr, irr=irrelevant_number, n=sample_size, types=generate_type, noise=0.05).fit()
    dataExp['Y'] = yExp
    filename = generate_type +'__' + str(irrelevant_number) + '__' + str(sample_size) + '.csv'
    path = './'
    dataExp.to_csv(path + filename, index=False)