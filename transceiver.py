import log
import sys
import math
import random
from collections import deque
import numpy as np
import soundfile as sf
import sounddevice as sd
import abc
import csv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Show silent exceptions caused by pyqt from matplotlib
def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


sys._excepthook = sys.excepthook
sys.excepthook = exception_hook


class named_deque(deque):
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)
        self.id = str(random.randint(0, 1000000000))


class BitOperations:
    def binary_repr(self, num, width):
        ret = np.binary_repr(num, width)
        if len(ret) != width:
            raise ValueError(f'Not enough width ({width}) to represent {num} in binary')
        return ret

    def bit_array_to_str(self, array):
        ret = ''
        for bit in array:
            ret += str(bit)
        return ret


class Signals:
    def __init__(self):
        self.sr = 44100
        self.sync_pulse_rendered = False
        self.delta_pulse_rendered = False
        self.bandwide_chirp_rendered = False

    def save_array_as_wav(self, file_name, array):
        log.debug(f"Saving {file_name} at sample rate {self.sr}")
        sf.write(file_name, array, self.sr)

    def normalize(self, sig):
        h = sig / np.linalg.norm(sig)
        return h

    def amplitude_modulate(self, sig, freq, m = 1):
        sin = self.get_sinewave(freq, len(sig))
        return (sig + m) * sin / (1 + m)

    def lowpass(self, sig, freq):
        N = len(sig)
        sig = np.fft.fft(sig)[:freq]
        sig = np.fft.ifft(sig)
        n = len(sig)
        ret = np.zeros(N)
        for i in range(N):
            ret[i] = sig[n * i//N]
        return ret

    def bias(self, sig):
        sig = sig.clip(0)
        return sig

    def mean_zero(self, sig):
        mean = np.mean(sig)
        return sig - mean

    def convolve(self, data, h):
        N = len(data)
        n = len(h)
        if N < n:
            raise ValueError("Filter longer than provided data")
        h = np.append(h, np.zeros(N-n))
        h = np.fft.rfft(h)
        data = np.fft.rfft(data)
        conv = h*data
        conv = np.fft.irfft(conv)
        assert len(conv) == N
        return conv

    def get_matched_filter(self, p):
        return self.normalize(p)[::-1]

    ############################################
    # CSV
    ############################################

    def get_csv_data(self, filename):
        data = []
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, )
            for row in spamreader:
                if len(row) == 1:
                    data.append(float(row[0]))
                elif len(row) == 0:
                    pass
                else:
                    log.warning(f"CSV reader - Length of row not 1: {row}")

        log.debug(f"Got data {filename}, shape {np.shape(data)}")
        return data

    def save_csv_data(self, filename, data):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([[i] for i in data])

    ############################################
    # Get functions
    ############################################

    def get_sinewave(self, freq, duration_samples):
        audio = np.zeros(duration_samples)
        for x in range(duration_samples):
            audio[x]=(math.sin(2 * math.pi * freq * (x / self.sr)))
        return audio

    def get_chirp(self, f1, f2, duration_samples):
        audio = np.zeros(duration_samples)
        for x in range(duration_samples):
            freq = (f1 + (f2 - f1) * x / duration_samples)
            audio[x] = (np.sin(math.pi * freq * (x / self.sr)))
        return audio

    def get_sync_pulse(self, f1=5000, f2=10000, duration_samples=2**12):
        audio = self.get_chirp(f1, f2,  duration_samples)

        if not self.sync_pulse_rendered:
            self.save_array_as_wav('sync.wav', audio)
            self.sync_pulse_rendered = True

        return audio

    def get_sync_pulse_matched_filter(self):
        return self.get_matched_filter(self.get_sync_pulse())

    def get_bandwide_chirp(self):
        sig = []
        sig.extend(self.get_sinewave(400, 2056))
        sig.extend(self.get_chirp(0, self.sr/2,  10*self.sr))
        sig.extend(self.get_sinewave(400, 2056))

        if not self.bandwide_chirp_rendered:
            self.save_array_as_wav('bandwide_chirp.wav', sig)
            self.bandwide_chirp_rendered = True

        return sig

    def get_channel_response(self,):
        return self.get_csv_data('impulse_response.csv')

    def get_raised_cosine(self, width_samples):
        pass

    def get_sinc_pulse(self, freq, duration_samples):
        pass


class Packet:
    """ Contains as bit arrays data to be transmitted, frames and order of frames"""

    def __init__(self, data):
        self.structure = ['data']
        self.data = {'data': data}

    def add_prefix_frame(self, name, frame_data):
        self.structure.insert(0, name)
        self.data[name] = frame_data

    def add_suffix_frame(self, name, frame_data):
        self.structure.append(name)
        self.data[name] = frame_data

    def add_sandwich_frame(self, name, pre_data, post_data):
        self.add_prefix_frame('pre_' + name, pre_data)
        self.add_suffix_frame('post_' + name, post_data)

    def unpack(self):
        return np.concatenate([self.data[name] for name in self.structure])


class Transceiver:
    def __init__(self):
        self.sig = Signals()

    pass
    # Contains items common to both the receiver and transmitter i.e. shape of sync pulse, common functions