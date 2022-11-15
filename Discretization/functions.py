from estimators import *

def entropy(Y):
    """
    H(Y)
    """
    #unique, count = np.unique(Y.astype('<U22'), return_counts=True, axis=0)
    entro =naive_estimate(Y)
    #prob = count/len(Y)
    #entro = np.sum((-1)*prob*np.log2(prob))
    return abs(entro)


def jEntropy(X, Y):
    """
    H(X,Y)
    """
    XY = np.c_[X, Y]
    return entropy(XY)


def cEntropy(Y, X):
    """
    H(Y|X) = H(Y,X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)


def gain(X, Y):
    """
    Information Gain, I(X;Y) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y, X)


def fGain(X, Y):
    """
    Fraction of information, F(X;Y) = I(X;Y)/H(Y)
    """
    return gain(X, Y) / entropy(Y)


def cab(a, b, result=1):
    """
    Function of C^a_b = a*...(a-b+1)/b!
    """
    if b == 0:
        return 1
    if b < 0:
        raise Exception("Expect boundary larger than 0")
    elif a < b:
        return 0
    while b > 0:
        result *= a / b
        a -= 1
        b -= 1
    return int(result)
