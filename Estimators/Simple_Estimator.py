from .Fingerprints import *
from .James_Stein import *
import math

class Simple_estimator:

    def __init__(self, data):
        # initial variables
        self.data = data
        finger = Fingerprints(data).get_fingers()
        self.fingerprints = finger
        self.length = len(finger)  # length of fingerprints
        self.size = sum([(i + 1) *finger[i] for i in range(self.length)])  # sample size

    def naive_estimate(self):
        """
        naive estimator
        """
        result = 0
        for i in range(self.length):
            result += self.fingerprints[i]*(i+1)/self.size * math.log((i+1)/self.size, 2)

        return -result

    def miller_estimate(self):
        """
        Miller-Madow
        """
        nai = self.naive_estimate()
        result = nai + (sum(self.fingerprints)-1)/(2*self.size)
        return result

    def coverge(self):
        """
        Coverage adjusted estimator
        """
        result = 0

        f1 = self.fingerprints[0]
        ps = 1 - f1 / self.size

        for i in range(self.length):
            fi = self.fingerprints[i]
            try:
                current = -fi * ((i + 1) / self.size) * ps * math.log((i + 1) / self.size * ps, 2) / (1 - (1 - (i + 1) / self.size * ps) ** self.size)
            except:
                current = 0
            result += current
        return result

    def jack(self):
        """
        Jackknifed naive estimator
        """
        current = 0
        n = len(self.data)

        for i in range(n):
            copy_data = self.data.copy()
            copy_data.pop(i)

            current_nai = Simple_estimator(copy_data).naive_estimate()
            print(copy_data, current_nai)
            current += current_nai

        nai = Simple_estimator(self.data).naive_estimate()

        result = n * nai - (n - 1) / n * current
        return result

    def james(self):
        """
        Jame stein setimator
        """
        result = James_Stein().James_estimate(self.data)
        return result
