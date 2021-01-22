import numpy as np


def equal_width_1d(data, k, a=None, b=None):
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
    >>> equal_width(np.array([[2.0, 0], [3.5, 0.5], [2.7, 5]]), 3)
    array([[0, 0], [2, 0], [1, 2]])
    """
    res = np.zeros_like(data, dtype='int')
    a = data.min() if a is None else a
    b = data.max() if b is None else b
    h = (b - a) / k
    for i in range(len(data)):
        res[i] = int((data[i] - a) // h) - (data[i] % h == 0 and data[i] != a)
        res[i] = k - 1 if res[i] >= k - 1 else res[i]
    return res


def equal_width(data, k, a=None, b=None):
    res = np.zeros_like(data, dtype='int')
    d = len(data[0]) if type(data[0]) in [np.array, np.ndarray, list] else 1
    for j in range(d):
        if d != 1:
            subdata = data[:, j] if d != 1 else data
            res[:, j] = equal_width_1d(subdata, k=k, a=a, b=b)
        else:
            res = equal_width_1d(data, k=k, a=a, b=b)
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
        >>> equal_freq(np.array([[2.0, 1], [3.5,2], [2.7,3]]), 2)
        array([[0, 0], [1, 0], [0, 1]])
        """
    d = len(data[0]) if type(data[0]) in [np.array, np.ndarray, list] else 1
    new_bins_list = np.zeros_like(data, dtype='int')
    for j in range(d):
        subdata = data[:, j] if d != 1 else data
        bins_dict = dict(zip([_ for _ in range(len(subdata))], subdata))
        bins_list = sorted(bins_dict.items(), key=lambda x: x[1])
        bins_list = [list(each) for each in bins_list]
        for i in range(len(subdata)):
            bins_list[i][1] = i + 1
        length = len(subdata) // k  # each length length + 1
        remain = len(subdata) - k * length
        result_list = [[] for _ in range(k)]
        l = 0
        for each in result_list:
            new_length = length + 1 if remain > 0 else length
            while new_length != 0:
                each.append(bins_list[l][0])
                l += 1
                new_length -= 1
            remain -= 1
        for i in range(len(result_list)):
            for each in result_list[i]:
                if d != 1:
                    new_bins_list[each][j] = i
                else:
                    new_bins_list[each] = i
    return new_bins_list

if __name__ == '__main__':
    import doctest
    doctest.testmod()
