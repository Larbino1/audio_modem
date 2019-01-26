import log
import sys
import math
import random
from collections import deque
import numpy as np
import soundfile as sf
import sounddevice as sd
import abc

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
    def __init__(self):
        super().__init__()
        self.id = str(random.randint(0,1000000000))


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

    def modulate(self, sig, freq, m = 1):
        sin = self.get_sinewave(freq, len(sig))
        return (sig + m) * sin / (1 + m)

    def lowpass(self, sig, freq):
        cutoff = len(sig)*freq//self.sr
        sig = np.fft.fft(sig)[:cutoff]
        return np.fft.ifft(sig)

    def bias(self, sig):
        sig = sig.clip(0)
        return sig

    def mean_zero(self, sig):
        mean = np.mean(sig)
        return sig - mean

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

    def get_sync_pulse(self):
        sig = []
        sig.extend(self.get_chirp(4000, 5000,  128))
        # sig.extend(self.get_sinewave(5000, 1024))

        if not self.sync_pulse_rendered:
            self.save_array_as_wav('sync.wav', sig)
            self.sync_pulse_rendered = True

        return sig

    def get_bandwide_chirp(self):
        sig = []
        sig.extend(self.get_sinewave(400, 2056))
        sig.extend(self.get_chirp(0, self.sr/2,  10*self.sr))
        sig.extend(self.get_sinewave(400, 2056))

        if not self.bandwide_chirp_rendered:
            self.save_array_as_wav('bandwide_chirp.wav', sig)
            self.bandwide_chirp_rendered = True

        return sig


class Transceiver:
    def __init__(self):
        self.sig = Signals()

    pass
    # Contains items common to both the receiver and transmitter i.e. shape of sync pulse, common functions