<<<<<<< HEAD
<<<<<<< HEAD
"""
This algorithm is for MAIN variable selection
"""
from VariableMI import *
import itertools
from Functions import *
import numpy as np

class VariableSelectionMI:

    def __init__(self, x, y, column_names, estimator=naive_estimate, xbin=False, ybin=False, bin_type='freq', x_bin=2,
                y_bin=2, permut=False, fraction=True, individual=True, interaction=False):
=======
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
import itertools
from Functions import *
from binning import *
from Permutation import *


class VariableSelectionMI:

    def __init__(self, x, y, column_names, estimator=naive_estimate, bin_nums=False, bin_type='freq',
                 individual=True, interaction=False, nums=2, top_features=10, permut=False, fraction=True):
<<<<<<< HEAD
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
        self.estimator = estimator
        self.x = x
        self.y = y
        self.bin_type = bin_type
<<<<<<< HEAD
<<<<<<< HEAD
        self.xbin = xbin
        self.ybin = ybin
        self.names = column_names
        self.interaction = interaction
        self.individual = individual
        self.x_bin = x_bin
        self.y_bin = y_bin
=======
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
        self.bin_nums = bin_nums
        self.names = column_names
        self.interaction = interaction
        self.individual = individual
        self.nums = nums
        self.num_features = top_features
<<<<<<< HEAD
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
        self.mi_list = []
        self.inter_list = []
        self.indi_variables = None
        self.inter_variables = None
        self.permut = permut
        self.fraction = fraction

<<<<<<< HEAD
<<<<<<< HEAD
    def summary(self):
        if self.individual:
            for i in range(len(self.names)):
                mi = VariableMI(np.array(self.x[:, i], dtype=float), self.y, estimator=self.estimator, xbin=self.xbin,
                                ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin, y_bin=self.y_bin,
                                permut=self.permut,  fraction=self.fraction).summary() \
                    if len(self.names) != 1 \
                    else VariableMI(self.x).summary()
                self.mi_list.append((mi, self.names[i]))
            self.mi_list.sort(reverse=True)

        all_interactions = list(itertools.combinations([_ for _ in range(len(self.x[0]))], self.x_bin))
        if self.interaction:
            for cell in all_interactions:
                int_mi = VariableMI(np.array(self.x[:, cell], dtype=float), self.y, estimator=self.estimator,
                                    xbin=self.xbin, ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin,
                                    y_bin=self.y_bin, permut=self.permut,fraction=self.fraction).summary()
                self.inter_list.append((int_mi, (self.names[cell[0]], self.names[cell[1]])))
            self.inter_list.sort(reverse=True)
        return self.mi_list, self.inter_list
=======
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
    def summary(self, scores=False):
        if self.individual:
            d = len(self.x[0]) if type(self.x[0]) in [np.array, np.ndarray, list] else 1
            for j in range(d):
                x_features = self.x[:, j] if d != 1 else self.x
                if self.bin_nums and self.bin_nums != 0:
                    x_freq = equal_freq(x_features, self.bin_nums) \
                        if self.bin_type == 'freq' \
                        else equal_width(x_features, self.bin_nums)
                    #y_freq = equal_freq(self.y, self.bin_nums) \
                    #    if self.bin_type == 'freq' \
                    #    else equal_width(self.y, self.bin_nums)
                else:
                    x_freq = x_features
                y_freq = self.y
                mi = fGain(x_freq, y_freq) if self.fraction else gain(x_freq, y_freq)
                if self.permut:
                    permut = Permutation(x_freq, y_freq).summary()
                    permut = permut/entropy(y_freq) if self.fraction else permut
                    mi -= permut
                self.mi_list.append((mi, str('X' + str(j))))
            self.mi_list.sort(reverse=True)
            if scores:
                self.indi_variables = [(each[0], int(each[1][1:])) for each in self.mi_list[:self.num_features]]
            else:
                self.indi_variables = [int(each[1][1:]) for each in self.mi_list[:self.num_features]]
        if self.interaction:
            all_interactions = list(itertools.combinations([_ for _ in range(len(self.x[0]))], self.nums))
            for cell in all_interactions:
                if self.bin_nums:
                    x_freq = equal_freq(self.x[:, cell], self.bin_nums) \
                        if self.bin_type == 'freq' \
                        else equal_width(self.x[:, cell], self.bin_nums)
                    y_freq = equal_freq(self.y, self.bin_nums) \
                        if self.bin_type == 'freq' \
                        else equal_freq(self.y, self.bin_nums)
                else:
                    x_freq = self.x[:, list(cell)]
                    y_freq = self.y
                gains = 0
                for each in cell:
                    gains += fGain(self.x[:, each], y_freq)
                entro = fGain(x_freq, y_freq) - gains
                if self.bin_nums and self.permut:
                    permut = Permutation(x_freq, y_freq).summary() / entropy(y_freq)
                    mi -= permut
                self.inter_list.append((entro, "-".join(list([str(each) for each in cell])))) \
                    if entro >= 0.05 \
                    else self.inter_list
            self.inter_list.sort(reverse=True)
            if scores:
                self.inter_variables = [(each[0], [int(sub) for sub in each[1].split('-')]) for each in
                                        self.inter_list[:self.num_features]]
            else:
                self.inter_variables = [[int(sub) for sub in each[1].split('-')] for each in
                                        self.inter_list[:self.num_features]]
        return self.indi_variables, self.inter_variables
<<<<<<< HEAD
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
=======
>>>>>>> f5a813a77e811307eafb2915efd86e9b6752a9df
