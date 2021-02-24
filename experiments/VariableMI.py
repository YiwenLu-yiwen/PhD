"""
This algorithm is for calculate single variable with target variable MI
"""
from binning import *
from Permutation import *


class VariableMI:

    def __init__(self, x, y, estimator=naive_estimate, xbin=False, ybin=False, bin_type='freq', x_bin=2, y_bin=2,
                 permut=False, fraction=True):
        self.estimator = estimator
        self.x = x
        self.y = y
        self.bin_type = bin_type
        self.xbin = xbin
        self.ybin = ybin
        self.x_bin = x_bin
        self.y_bin = y_bin
        self.permut = permut
        self.fraction = fraction

    def summary(self):
        x_features = self.x
        if self.xbin and self.x_bin != 0:   # x bins
            x_freq = equal_freq(x_features, self.x_bin) \
                if self.bin_type == 'freq' \
                else equal_width(x_features, self.x_bin)
        else:
            x_freq = x_features
        if self.ybin and self.y_bin != 0:  # y bins
            y_freq = equal_freq(self.y, self.y_bin) \
                if self.bin_type == 'freq' \
                else equal_width(self.y, self.y_bin)
        else:
            y_freq = self.y
        mi = fGain(x_freq, y_freq) if self.fraction else gain(x_freq, y_freq)
        if self.permut:
            permut = Permutation(x_freq, y_freq).summary()
            permut = permut / entropy(y_freq) if self.fraction else permut
            mi -= permut
        return mi
