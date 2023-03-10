# 01/25/23 modified functions to class
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from copy import deepcopy
import numpy as np
import pandas as pd
import math
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

seeds = 1000

class linear_base_model:
    """This algorithm based on linear regression and variable selection is based on the p values.
    We set if |p < 0.05|, it is significant
    Input:
        X: predictors
        y: targets
    Output:
        variable index list: for example X_1, X_2 will covert to [0, 1]
    """
    def __init__(self) -> None:
        self.best_features = []
        pass

    def fit(self, X, y):
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
        self.best_features = ['X' + str(i-1) for i in range(len(p_values)) if p_values[i] < 0.05 and i != 0]
        return None, self.best_features

class logistic_forward_selection:
    """
    This algorithm is for logistic regression forward selection with AIC values
    """
    def __init__(self) -> None:
        self.best_features = []
        self.best_aic = None
        pass

    def fit(self, X, y):
        initial_features = X.columns.tolist()
        initial_features.sort()
        best_features = []
        best_aic = None
        initial_features = X.columns.tolist()
        best_features = []
        while (len(initial_features)>0):
            remaining_features = list(set(initial_features)-set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                try:
                    model = GLM(y, sm.add_constant(X[best_features+[new_column]]), family=sm.families.Binomial()).fit()
                    new_pval[new_column] = model.aic
                except:
                    new_pval[new_column] = len(best_features+[new_column])/2 * math.log(len(y)) # this is model bic
            min_aic_value = new_pval.min()

            if(best_aic == None or best_aic > min_aic_value):
                best_features.append(new_pval.idxmin())
                best_aic = min_aic_value
            else:
                break
            best_features.sort()
            if best_features == initial_features:
                break
        self.best_features, self.best_aic = best_features, best_aic
        return best_aic, best_features


class lasso_base_model:

    """This algorithm use lasso regression and variable selection is based on the coeffient.
    Input:
        X: predictors, need dataframes
        y: targets
    Output:
        variable index list: for example X_1, X_2 will covert to [0, 1]
    """
    def __init__(self, base_model=LogisticRegressionCV(penalty='l1', solver='saga', random_state=seeds)) -> None:
        self.base_model = base_model

    def fit(self, X, y):
        self.base_model.fit(X, y)
        # fit lasso
        return None, [X.columns[i] for i in range(len(self.base_model.coef_[0])) if self.base_model.coef_[0][i] != 0 and i != 0]

class random_forest:
    """
    Use feature importance to get the variables
    """
    def __init__(self, base_model=RandomForestClassifier(random_state=seeds)) -> None:
        self.base_model = base_model
        self.best_subset = None

    def fit(self, X, y):
        # 5 folder cross validation
        random_forest_base = self.base_model#, min_samples_leaf=1, n_estimators=200)
        random_forest_base_copy = deepcopy(random_forest_base)
        random_forest_base_copy.fit(X, y)
        rf_imp=random_forest_base_copy.feature_importances_
        cols = X.columns
        rk_dict = dict(zip(cols, rf_imp))
        rk_dict = {k:v for k, v in rk_dict.items() if v} # filter non-zero feature importance
        ranking_variables = list(rk_dict.keys()) # lowest feature importance to highest feature importance
        return None,  ranking_variables

    

class random_forest_wrapper:
    """This is random forest selection with entropy
    Backward elimation with performance comparison
    """

    def __init__(self, base_model=RandomForestClassifier(random_state=seeds)) -> None:
        self.base_model = base_model
        self.best_subset = None

    def cross_validation_model(self, model, X, y, folds=5):
        """This algorithm applies cross validation to models
        """
        error = 0
        for train_idx, test_idx in KFold(n_splits=folds).split(X, y):
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model_copy = deepcopy(model)
            model_copy.fit(x_train, y_train)
            err =  sum((y_test - model_copy.predict(x_test))**2)
            error += err/folds
        return error
    
    def fit(self, X, y):
        # 5 folder cross validation
        # backward elimation
        random_forest_base = self.base_model#, min_samples_leaf=1, n_estimators=200)
        # random_forest_base_copy = deepcopy(random_forest_base)
        # random_forest_base_copy.fit(X, y)
        # rf_imp=random_forest_base_copy.feature_importances_
        # cols = X.columns
        # rk_dict = dict(zip(cols, rf_imp))
        # rk_dict = {k: v for k, v in sorted(rk_dict.items(), key=lambda item: item[1], reverse=False)}
        # ranking_variables = list(rk_dict.keys()) # lowest feature importance to highest feature importance
        ranking_variables = X.columns
        stop = []
        best_performance = self.cross_validation_model(deepcopy(random_forest_base), X, y)
        best_X, candidate_X = deepcopy(X), deepcopy(X)
        # backward elimation based on variable ranking
        while not stop:
            for variable in ranking_variables:
                X_copy = deepcopy(best_X)
                X_copy.drop(columns=[variable])
                current_performance = self.cross_validation_model(deepcopy(random_forest_base), X_copy, y)
                if current_performance < best_performance or (current_performance == best_performance and len(X_copy.columns) < len(best_X.columns)):
                    print(current_performance, best_performance)  
                    best_performance = current_performance
                    candidate_X = deepcopy(X_copy)  
                                
            if list(best_X.columns) == list(candidate_X.columns):
                stop = True
            else:
                best_X = deepcopy(candidate_X)
                ranking_variables = candidate_X.columns
        self.best_subset = best_X.columns.tolist()
        return None, self.best_subset

