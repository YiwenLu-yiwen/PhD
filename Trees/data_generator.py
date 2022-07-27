import numpy as np
from scipy.stats import norm, entropy, uniform, bernoulli
from scipy.special import expit
import pandas as pd
import sklearn.datasets as dt
from sklearn import cluster

RNG = np.random.default_rng(seed = 0)

var2 = 0.5
marginal_x1_pdf = uniform(-8, 8).pdf # norm(0, 4).pdf  

def cond_mean_x2(x1):
    return x1+2*np.sin(10*x1/(2*np.pi))

# generate data
# def rvs(n, irr=100):
#     x1 = norm(0, 1).rvs(size=n)
#     x2 = norm(0, 1).rvs(size=n)
#     x3 = norm(0, 1).rvs(size=n)

#     y = bernoulli.rvs(expit(x1+x2+x3), random_state=RNG)
    
#     irr_lst = [uniform(-2, 2).rvs(n) for _ in range(irr)]
#     for each in [x1, x2, x3]:
#         irr_lst.append(each)
#     irr_columns = ['X' + str(i) for i in range(4, irr+4)]
#     rr_columns = ['X1', 'X2', 'X3']
#     cols = irr_columns + rr_columns
#     df = pd.DataFrame(np.column_stack(irr_lst))
#     df.columns = cols
#     return df, y
class dataGenerator2d:
    def __init__(self, n, irr, types='independent_circle', factor=0.5, noise=0.05) -> None:
        self.n = n
        self.irr = irr
        self.types = types
        self.factor = factor
        self.noise = noise
    
    def fit(self):
        if self.types == 'independent_circle':
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
        elif self.types == 'no structure':
            x1 = norm(0, 1).rvs(size=self.n)
            x2 = norm(0, 1).rvs(size=self.n)
            y = bernoulli.rvs(expit(x1-x2), random_state=RNG)
        
        irr_lst = [norm(0, 1).rvs(size=self.n) for _ in range(self.irr)]
        for each in [x1, x2]:
            irr_lst.append(each)
        irr_columns = ['X' + str(i) for i in range(2, self.irr+2)]
        rr_columns = ['X0', 'X1']
        cols = irr_columns + rr_columns
        df = pd.DataFrame(np.column_stack(irr_lst))
        df.columns = cols
        return df, y

def rvs2d(n, irr, types='independent_circle', factor=0.5, noise=0.05):
    if types == 'independent_circle':
        X, y = dt.make_circles(n_samples=n, factor=factor, noise=noise)
    elif types == 'mixed_circle':
        X, y = dt.make_circles(n_samples=n, factor=factor, noise=noise)
        params = {'n_neighbors': 10, 'n_clusters': 2}
        average = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="average"
    )
        y = average.fit_predict(X)
    elif types == 'no structure':
        x1 = norm(0, 1).rvs(size=n)
        x2 = norm(0, 1).rvs(size=n)
        y = bernoulli.rvs(expit(x1-x2), random_state=RNG)
    
    irr_lst = [norm(0, 1).rvs(size=n) for _ in range(irr)]
    for each in [x1, x2]:
        irr_lst.append(each)
    irr_columns = ['X' + str(i) for i in range(2, irr+2)]
    rr_columns = ['X0', 'X1']
    cols = irr_columns + rr_columns
    df = pd.DataFrame(np.column_stack(irr_lst))
    df.columns = cols
    return df, y

# test1: 2d all independent: N(0,1), N(0,1)

# test2

# test3

# test4