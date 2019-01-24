import log
import math
import abc
import sys
import queue
from collections import deque
import threading
import multiprocessing
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Show silent exceptions caused by pyqt from matplotlib
def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


sys._excepthook = sys.excepthook
sys.excepthook = exception_hook

import numpy as np
import random

import sounddevice as sd
import soundfile as sf

AUDIO_PLOT_LENGTH = 500 * 44100 // 1000

class Signals:
    def __init__(self):
        self.sr = 44100
        self.sync_pulse_rendered = False
        self.delta_pulse_rendered = False
        self.bandwide_chirp_rendered = False

    def get_sinewave(self, freq, duration_samples):
        audio = []
        for x in range(duration_samples):
            audio.append(math.sin(2 * math.pi * freq * (x / self.sr)))
        return audio

    def get_chirp(self, f1, f2, duration_samples):
        audio = []
        for x in range(duration_samples):
            freq = (f1 + (f2 - f1) * x / duration_samples)
            audio.append(np.sin(math.pi * freq * (x / self.sr)))
        return audio

    def get_sync_pulse(self):
        sig = []
        # sig.extend(self.get_chirp(400, 4000,  5000))
        sig.extend(self.get_sinewave(5000, self.sr//10))

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

    def get_delta(self):
        sig = [1]

        if not self.delta_pulse_rendered:
            self.save_array_as_wav('delta.wav', sig)
            self.sync_pulse_rendered = True

        return sig

    def normalize(self, sig):
        h = sig / np.linalg.norm(sig)
        return h

    def save_array_as_wav(self, file_name, array):
        log.debug(f"Saving {file_name} at sample rate {self.sr}")
        sf.write(file_name, array, self.sr)


class Transceiver:
    def __init__(self):
        self.sig = Signals()

    pass
    # Contains items common to both the receiver and transmitter i.e. shape of sync pulse, common functions


class Modulator:
    def __init__(self):
        pass

    @abc.abstractmethod
    def modulate(self, data_bit_array):
        """
        Receives an array of bits and returns an audio array
        Also adds any modulation specific audio such as calibration symbols etc
        """


class Demodulator:
    def __init__(self):
        pass

    @abc.abstractmethod
    def demodulate(self, audio_array):
        """
        Recevies an audio array and returns an array of bits
        """


class PAM(Modulator):
    pass


class Receiver(Transceiver):
    # Has specific log color/tag
    def __init__(self):
        Transceiver.__init__(self)
        self.blocksize = 1024
        self.q = multiprocessing.Queue()
        self.recording_flag = False
        self.stream = sd.InputStream(channels=1, samplerate=self.sig.sr, callback=self.audio_callback, blocksize=self.blocksize)
        self.alloc_thread = None
        self.scan_thread = None
        self.plotdata = []

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        # Fancy indexing with mapping creates a (necessary!) copy:
        # print(f"put {indata[:]}")
        if status:
            log.error('audio_callback error:')
            log.error(status)
        self.q.put(indata[:])

    def record(self, destinations):
        log.info("STARTED RECORDING")
        self.stream.start()
        self.recording_flag = True
        self.alloc_thread = threading.Thread(target=self.allocator, args=[destinations])
        self.alloc_thread.start()

    def allocator(self, destinations):
        log.debug('STARTING ALLOCATING')
        while self.recording_flag:
            try:
                item = self.q.get()
                # print(f"got {item}")
                for target in destinations:
                    target.append(item)
            except queue.Empty:
                time.sleep(0.2)
            if self.q.qsize() > 128:
                log.warning(f'Recording queue backing up, qsize {self.q.qsize()}')
        log.debug('STOPPING ALLOCATING')

    def stop(self):
        log.info("STOPPED RECORDING")
        self.stream.stop()
        self.recording_flag = False
        self.alloc_thread.join(timeout=1)
        self.scan_thread.join(timeout=1)

    def scan_queue(self, data_queue, output_queue, h):
        self.scan_thread = threading.Thread(target=self.scanner, args=(data_queue, output_queue, h))
        self.scan_thread.start()

    def scanner(self, data_queue, output_queue, h):
        while self.recording_flag:
            try:
                 data = data_queue.popleft()
                 output_queue.append(r.convolve(data, h))
            except IndexError:
                time.sleep(0.1)
        # new_data = r.convolve(np.concatenate(scan_plot_data[-len(h):], data), h)[len(h):]
        # scan_plot_data[-shift:, :] = new_data

    def show(self, data_queue, figax=None, show=True):
        log.info('Showing audio')

        if figax:
            fig, ax = figax
        else:
            fig, ax = plt.subplots(nrows=1,)

        global showplotdata
        showplotdata = np.zeros((AUDIO_PLOT_LENGTH, 1))

        ax.axis((0, len(showplotdata), -1, 1))
        lines = ax.plot(showplotdata)

        def update_plot(frame):
            """This is called by matplotlib for each plot update.

            Typically, audio callbacks happen more frequently than plot updates,
            therefore the queue tends to contain multiple blocks of audio data.
            """
            global showplotdata
            while True:
                try:
                    data = data_queue.popleft()
                except IndexError:
                    break
                shift = len(data)
                showplotdata = np.roll(showplotdata, -shift, axis=0)
                showplotdata[-shift:, :] = data

            for column, line in enumerate(lines):
                line.set_ydata(showplotdata[:, column])
            return lines

        ani = FuncAnimation(fig, update_plot, interval=30, blit=True)
        if show:
            plt.show()
        log.info('Stopping showing audio')

    def collapse_queue(self, q):
        l = np.array(q)
        l = np.concatenate(l)
        l = np.transpose(l)
        l = l[0]
        return l

    def convolve(self, data, h):
        N = len(data)
        n = len(h)
        log.debug(f"h shape {np.shape(h)}")
        log.debug(f"data shape {np.shape(data)}")

        if N < n:
            raise ValueError
        h = np.append(h, np.zeros(N-n))
        h = np.fft.rfft(h)
        data = np.fft.rfft(data)
        conv = h*data
        conv = np.fft.irfft(conv)
        log.debug(f"Convolved to shape {np.shape(conv)}")
        return conv

    # Process queue making sure to have overlap between chunks, identify given signal
    # To identify convolve with reversed signal filter, detect if over threshold, locate peak location

    # If signal is identified save queue from peak for specified length of time to array


class Transmitter(Transceiver):

    def __init__(self):
        Transceiver.__init__(self)

    # Has log colour/tag
    pass

    def play_wav(self, file_name):
        data, fs = sf.read(file_name)
        sd.play(data, fs)

    # Produce audio clip with leading silence and synchronisation signal

    # Play audio clip on speaker


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


def initialise():
    r = Receiver()
    t = Transmitter()
    return r, t


# if __name__ == '__main__':

q1 = deque()
q2 = deque()
q3 = deque()

log.info('MAIN')
r, t = initialise()

h = r.sig.normalize(r.sig.get_sync_pulse())

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


fig, ax = plt.subplots(nrows=2, sharex='all')
ax0 = ax[0]
ax1 = ax[1]
r.record([q1, q2])

h = [0,1,0,1]
r.scan_queue(q2, q3, h)

# r.show(q3, (fig, ax0), show=True)
r.show(q3, (fig, ax1), show=True)
plt.show()


r.stop()
