import numpy as np
from scipy.stats import norm, entropy, uniform, bernoulli
from sklearn.datasets import make_classification
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
    def __init__(self, n, irr, rr=2, types='independent_circle', factor=0.5, noise=0.05) -> None:
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

        elif self.types == 'no_structure':
            x1 = norm(0, 1).rvs(size=self.n)
            x2 = norm(0, 1).rvs(size=self.n)
            y = bernoulli.rvs(expit(x1-x2))

        elif self.types == 'interaction':
            x1 = norm(0, 1).rvs(size=self.n)
            x2 = norm(0, 1).rvs(size=self.n)
            x3 = x1 * x2
            y = bernoulli.rvs(expit(x1-x2+x3))
        
        elif self.types == 'make_classfication':
            n_features = self.rr + self.irr
            X, y = make_classification(n_samples=self.n, n_features=n_features, n_informative=self.rr, n_redundant=0, n_repeated=0, n_classes=2, 
                    n_clusters_per_class=2, weights=None, flip_y=self.noise, class_sep=1.0, hypercube=True, shift=0.0, 
                    scale=1.0, shuffle=False, random_state=None)
            df = pd.DataFrame(X, columns=['X'+str(i) for i in range(X.shape[1])])
            return df, y
        
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
            for each in [x2, x1]:
                irr_lst.insert(0, each)
            irr_columns = ['X' + str(i) for i in range(2, self.irr+2)]
            rr_columns = ['X0', 'X1']
            cols = rr_columns + irr_columns
            df = pd.DataFrame(np.column_stack(irr_lst))
            df.columns = cols
            return df, y

        

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