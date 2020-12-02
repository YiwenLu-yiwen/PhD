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
        res[i] = k-1 if res[i] >= k-1 else res[i]
    return res

def equal_freq(data, k):
    """Partition numerical data into bins of equal frenquency.
        :param array data: data to be discretized
        :param int k: nuber of bins
        :returns: array of dtype int corresponding to binning
        For example:
        >>> equal_freq(np.array([0, 0.5, 2, 5, 10]), 3)
        array([0, 0, 1, 1, 2])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 1)
        array([0, 0, 0])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 2)
        array([0, 1, 0])
        """

    bins_list = []
    n = len(data)
    sub = []  # record sublist values
    datalist = sorted(data)
    i = 0
    result = [0 for _ in range(n)]
    while i <= len(datalist):
        if len(sub) >= int(n / k) and len(bins_list) != k - 1:
            bins_list.append((min(sub), max(sub)))  # record minimum and maximum
            sub = []
        if i == len(datalist):
            bins_list.append((min(sub), max(sub)))  # record minimum and maximum
            break
        sub.append(datalist[i])
        i += 1

    for j in range(k):
        for l in range(n):
            if data[l] >= min(bins_list[j]) and data[l] <= max(bins_list[j]):
                result[l] = j
    return result

if __name__=='__main__':
    import doctest
    doctest.testmod()

