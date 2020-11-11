import numpy as np

def equal_width(data, k, a=None, b=None):
    """Partition numerical data into bins of equal width.
    :param array data: data to be discretized
    :param int k: nuber of bins
    :param int|float a: virtual min value of partioned interval (default data.min())
    :param int|float b: virtual max value of partioned inverval (default data.max())
    :returns: array of dtype int corresponding to binning
    For example:
    >>> equal_width(np.array([0, 0.5, 2, 5, 10]), 10)
    array([0, 0, 1, 4, 9])
    >>> equal_width(np.array([2.0, 3.5, 2.7]), 3)
    array([0, 2, 1])
    >>> equal_width(np.array([2.0, 3.5, 2.7]), 5, a=0, b=5)
    array([1, 3, 2])
    """
    a = data.min() if a is None else a
    b = data.max() if b is None else b
    h = (b-a)/k
    res = np.zeros_like(data, dtype='int')
    for i in range(len(data)):
        res[i] = int((data[i] - a) // h) - (data[i] % h == 0 and data[i]!=a)
    return res

def equal_freq(data, k):
    """Partition numerical data into bins of equal frenquency.
        :param array data: data to be discretized
        :param int k: nuber of bins
        :returns: array of dtype int corresponding to binning
        For example:
        >>> equal_freq(np.array([0, 0.5, 2, 5, 10]), 3)
        array([0, 1, 2, 2, 2])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 1)
        array([0, 0, 0])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 2)
        array([0, 1, 1])
        """
    index = np.argsort(data)
    res = np.zeros_like(index, dtype='int')
    h = int(len(data)/k)
    for i in range(len(data)):
        res[i] = index[i] // h if index[i] // h < k else k-1
    return res


if __name__=='__main__':
    import doctest
    doctest.testmod()

