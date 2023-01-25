"""This script is applied early stopping
"""
from scipy.stats.distributions import chi2


def match_stopping(current_value, previous_value, size=None, degree_of_freedom=None, delta=None, typ=None):
    """Combine all the stopping criteria
    chi_square:
        degree_of_freedom = num_of_parameters - 1
        current_values, previous_value: loglikelihood. For example, mutual information is normalized likelihood
                                        which needs to convert back respect to their size.
    """
    if typ =='chi_square' or typ == 'chi_square_adjust':
        return chi_square_stopping(current_value, previous_value, size, degree_of_freedom, delta)
    else:
        return lazy_stopping(current_value, previous_value)

def chi_square_stopping(current_value, previous_value, size, degree_of_freedom, delta=0.05):
    """
    This algorithm is used non-normalized values
    return True if values are improved
    return False if values are not improved
    There is assumption that E(current_value - previous_value) is postive and small
    """
    if degree_of_freedom < 0:
        return True
    p_value = 1 - chi2.cdf(2*(current_value - previous_value)*size, degree_of_freedom)
    return p_value <= delta

def lazy_stopping(current_value, previous_value):
    """
    the easiest stopping criteria
    return True if values are improved
    return False if values are not improved
    """
    return current_value > previous_value