import math

class Binning:
    def __init__(self, datalist):
        self.data = sorted(datalist)
        self.size = len(datalist)
        self.bin = int(math.log(self.size, 2))

    def equal_freq(self):
        sub = []
        bins_list = []
        i = 0
        while i <= len(self.data):
            if len(sub) >= int(self.size/ self.bin) and len(bins_list) != self.bin - 1:
                bins_list.append((min(sub), max(sub)))  # record minimum and maximum
                sub = []

            if i == len(self.data):
                bins_list.append((min(sub), max(sub)))  # record minimum and maximum
                break

            sub.append(self.data[i])

            i += 1
        return bins_list

    def equal_width(self):
        bins_list = []
        bins = self.bin

        cnt = (self.data[-1] - self.data[0])/ bins
        pre = self.data[0]

        while bins > 0:
            bins_list.append((pre, pre + cnt))  # store values
            pre += cnt
            bins -= 1

        return bins_list


