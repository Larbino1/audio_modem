import log
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_csv_data(filename):
    data = []
    with open('freq_response.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile,)
        for row in spamreader:
            if len(row) == 1:
                data.append(float(row[0]))
            else:
                log.warning("Length of row not 1!")

    log.info(f"Got data {filename}, shape {np.shape(data)}")
    return data


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


data = get_csv_data('freq_response.csv')
env = get_envelope(data, 1000)
env = smooth_signal(env, 10)
x = np.linspace(0, 22000, 1000)
plt.plot(x, env)
plt.show()
