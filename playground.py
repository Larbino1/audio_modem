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

# SOMETHING FOR GETTING FREQ RESPONSE FROM CHIRP DATA
# r.record([q1])
# time.sleep(0.75)
# t.play_wav('sync.wav')
# time.sleep(1)
# r.stop()

# data = r.collapse_queue(q1)
# subsampling = 10
# plt.subplot(211)
# x=np.linspace(0, len(data)/r.sig.sr, len(data[::subsampling]))
# plt.plot(x, data[::subsampling])
#
# conv = r.convolve(data, h)
# plt.subplot(212)
# plt.plot(conv)
#
# plt.show()

# SOMETHING FOR GETTING FREQ RESPONSE FROM CSV
data = get_csv_data('recorded_long_chirp.csv')
env = get_envelope(data, 100)
env = smooth_signal(env, 10)
x = np.linspace(0, 22000, 100)
plt.plot(x, env)
plt.show()
save_csv_data('freq_response.csv', env)

freq_response = get_csv_data('freq_response.csv')
impulse_response = np.fft.irfft(freq_response)
impulse_response = impulse_response[:len(impulse_response)//2]
plt.plot(impulse_response)
plt.show()
save_csv_data('impulse_response.csv', impulse_response)
