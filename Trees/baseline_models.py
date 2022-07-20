from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy import stats
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

def linear_base_model(X, y):
    """This algorithm based on linear regression and variable selection is based on the p values.
    We set if |p < 0.05|, it is significant
    Input:
        X: predictors
        y: targets
    Output:
        variable index list: for example X_1, X_2 will covert to [0, 1]
    """
    # fit linear model
    lm = LogisticRegression()
    lm.fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    # get mse
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    # get variance, std, critical values (ts_b)
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    # based on t distribution
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    return [i-1 for i in range(len(p_values)) if p_values[i] < 0.05 and i != 0]

def forward_selection(data, target):
    """This algorithm is for logistic regression forward selection
    """
    initial_features = data.columns.tolist()
    initial_features.sort()
    best_features = []
    best_aic = None
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            try:
                model = GLM(target, sm.add_constant(data[best_features+[new_column]]), family=sm.families.Binomial()).fit()
                new_pval[new_column] = model.aic
            except:
                new_pval[new_column] = len(best_features+[new_column])/2 * math.log(len(target)) # this is model bic
        min_aic_value = new_pval.min()

        if(best_aic == None or best_aic > min_aic_value):
            best_features.append(new_pval.idxmin())
            best_aic = min_aic_value
        else:
            break
        best_features.sort()
        if best_features == initial_features:
            break
    return best_features, best_aic

def lasso_base_model(X, y):
    """This algorithm use lasso regression and variable selection is based on the coeffient.
    Input:
        X: predictors, need dataframes
        y: targets
    Output:
        variable index list: for example X_1, X_2 will covert to [0, 1]
    """
    lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', random_state=0)
    lasso.fit(X, y)
    # fit lasso
    return [X.columns[i] for i in range(len(lasso.coef_[0])) if lasso.coef_[0][i] != 0 and i != 0]
