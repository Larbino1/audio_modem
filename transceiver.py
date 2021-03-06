import log
import sys
import math
import random
from collections import deque
import numpy as np
import scipy.signal as scipy_sig
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

    def text_to_bits(self, text: str):
        send_bytes = list(text.encode())
        log.special(f'send_bytes {send_bytes}')
        return np.unpackbits(np.array(send_bytes, dtype='uint8'))


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

    def amplitude_modulate(self, sig, freq, m=1, auto_phase=False):
        sin = self.get_sinewave(freq, len(sig))
        if not auto_phase:
            ret = (sig + m) * sin / (1 + m)
        else:
            log.warning('Warning - be careful using auto_phase in amplitude_modulate, not safe')
            N = round(self.sr//freq)
            ret = []
            for i in range(N):
                ret.append((sig + m) * sin / (1 + m))
                np.roll(sin, 1)
            i = np.argmax([np.linalg.norm(x) for x in ret])
            ret = ret[i]
        assert len(ret) == len(sig)
        return ret

    def lowpass(self, sig, freq):
        sos = scipy_sig.butter(5, freq/self.sr, 'lp', output='sos')
        filtered = scipy_sig.sosfilt(sos, sig)
        return filtered

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


    def get_root_raised_cosine(self, T, b, width=5):
        """
        Gets x and y values for raised-cosine function with T and b parameters (domain is either 'time' or
         'frequency' ('time' by default). Gives root raised-cosine function in time domain
        """
        # Initialising axes
        x = np.append(np.linspace(0, np.pi/2, T*width/2), np.linspace(-np.pi/2, 0, T*width/2))
        y = []
        # if domain == 'frequency':

        # Raised-cosine in frequency domain
        thresh1 = (np.pi*(1 - b) / 2)*2/T
        thresh2 = (thresh1 + b*np.pi)*2/T
        for i in x:
            # log.special(i)
            if abs(i) <= thresh1:
                y.append(1)
            elif abs(i) <= thresh2:
                y.append(np.sqrt(0.5*(1 + np.cos((abs(i*T/2) - thresh1)/b))))
                # y.append(0.5*(1 + np.cos((abs(i*T/2) - thresh1)/b)))
            else:
                y.append(0)
        ifft = np.fft.ifft(y)
        pulse = np.roll(ifft, len(ifft) // 2)
        return pulse
        # y = np.fft.ifft(y)

        # Jeroen's work
        # elif domain == 'time':
        #     # Root raised-cosine in time domain
        #     for i in x:
        #         # Function split in parts for readability
        #         # Function defined in data transmission handout 2, page 26
        #         A = np.cos((1 + b) * np.pi * (i / T))
        #         d = (1 - b) * np.pi * (i / T)  # Intermediate for sinc part
        #         B = ((1 - b) * np.pi / (4 * b)) * np.sin(d) / d
        #         C = 1 - (4 * b * (i / T)) ** 2
        #         D = (4 * b) / (np.pi * np.sqrt(T))
        #         y.append(D * ((A + B) / C))
        # return x, y

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
        self.bop = BitOperations()

        self.debug_mode = True

        self.channels = {
            'ch1': {
                'freq': self.sig.sr/8,
            },
            'ch2': {
                'freq': self.sig.sr/9,
            },
            'ch3': {
                'freq': self.sig.sr/10,
            },
            'ch4': {
                'freq': self.sig.sr/11,
            },
        }

        self.defaults = {
            'audio_block_size': 2**12,
            'ampam': {
                # Pulse width/count for delivering actual pw/pc
                'initial_pulse_width': 1024,
                'threshold_data_bits': 4,
                'pulse_width_data_bits': 16,
                'pulse_count_data_bits': 16,
            },
            'qam': {
                'len': 0,
            }
        }
    # Contains items common to both the receiver and transmitter i.e. shape of sync pulse, common functions, default settings
