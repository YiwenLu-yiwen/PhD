class Fingerprints:

    def __init__(self, data):
        self.data = data

    def get_fingers(self):
        """
        get fingerprints
        """
        curr = 0
        i = 0
        self.data = sorted(self.data)  # sort data
        fingerprints = [0] * len(self.data)
        while i < len(self.data):
            if i == len(self.data) - 1:
                fingerprints[curr] += 1

            elif self.data[i] != self.data[i + 1]:
                fingerprints[curr] += 1
                curr = 0
            else:
                curr += 1
            i += 1

        return fingerprints
