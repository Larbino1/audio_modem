import log
import csv
import numpy as np
import matplotlib.pyplot as plt


def get_csv_data(filename):
    data = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile,)
        for row in spamreader:
            if len(row) == 1:
                data.append(float(row[0]))
            elif len(row) == 0:
                pass
            else:
                log.warning(f"CSV reader - Length of row not 1: {row}")

    log.info(f"Got data {filename}, shape {np.shape(data)}")
    return data


def save_csv_data(filename, data):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[i] for i in data])


def get_envelope(data, bin_count):
    """ Returns the envelope of a set of data"""
    bin_width = len(data) // bin_count
    mags = [0] * bin_count
    for i in range(bin_count):
        for val in data[i * bin_width: (i + 1) * bin_width]:
            if abs(val) > mags[i]:
                mags[i] = abs(val)
    return mags


def smooth_signal(sig, sigma):
    for i in range(sigma):
        sig = np.convolve(sig,[0.5, 0.7071, 0.5], mode='same')
    return sig


from transceiver import *
sig = Signals()
p = sig.get_root_raised_cosine(10, 0, width=2)
plt.plot(p)
plt.show()
