import math
class James_Stein:
    def __init__(self):
        pass

    def thetak(self, k, n):
        """
        define maximum likelihood theta_k for multi-nomial distribution
        """
        return k / n

    def sum_estimator(self, dic, p, n, tk):
        """
        this function is used to calculate shrinkage intensity
        result: sum(t_k - \theta)^2 or sum(\theta)^2
        """
        result = 0
        for each in dic.keys():
            k = dic[each]
            result += (tk - self.thetak(k, n)) ** 2
        return result

    def get_lambda(self, dic, p, n):
        """
        get James Shrinkage intensity: lambda
        """
        sub = self.sum_estimator(dic, p, n, 1 / p)
        if n == 1 or sub == 0:
            return 1
        lam = (1 - self.sum_estimator(dic, p, n, 0)) / ((n) * sub)
        return lam

    def James_estimator(self, dic, k, p, n):
        """
        get James Shrinkage estimator
        return: \theta_{shrinkage}
        """
        _lambda = self.get_lambda(dic, p, n)
        if _lambda > 1:
            _lambda = 1
        elif _lambda < 0:
            _lambda = 0
        result = _lambda / p + (1 - _lambda) * self.thetak(k, n)
        return result

    def James_estimate(self, alist):
        """
        This part is to get James Shrinkage entropy estimatation
        return: \hat H^{shrink}
        """

        population = list(set(alist))
        dic = dict(zip(population, [0] * len(population)))

        # indecate some variables
        n = len(alist)  # n
        for each in alist:
            dic[each] += 1

        p = len(population)  # p

        result = 0
        for each in dic.keys():
            k = dic[each]  # k
            if k == 0:
                continue
            # a = get_lambda(dic, p, n) # check lambda

            # if lambda not in [0,1], print('Error Message')
            # if a < 0 or a > 1:
            #    print('James Shrinkage Error Lambda:', a, dic, p, n, k)
            shrink_estimator = self.James_estimator(dic, k, p, n)

            # estimate result
            try:
                result += -shrink_estimator * math.log(shrink_estimator, 2)
            except:
                # return shrink_estimator
                return ('Error happening: ', dic, k, p, n)
        # print('James Shrinkage entropy estimate: ', result)
        return result