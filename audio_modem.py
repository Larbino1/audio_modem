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

AUDIO_PLOT_LENGTH = 3 * 500 * 44100 // 1000


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
        sig = np.fft.fft(sig)[:500]
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
        # sig.extend(self.get_chirp(400, 4000,  5000))
        sig.extend(self.get_sinewave(5000, 1024))

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
        self.q = multiprocessing.Queue()
        self.blocksize = 2 ** 10
        self.recording_flag = False
        self.stream = sd.InputStream(channels=1, samplerate=self.sig.sr, callback=self.audio_callback, blocksize=self.blocksize)
        self.alloc_thread = None
        self.scan_thread = None
        self.ani = []
        self.plotdata = dict()

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
                    target.append(np.concatenate(item))
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
        n = len(h)
        data_old = np.zeros(n)
        while self.recording_flag:
            try:
                data_new = data_queue.popleft()
                output_queue.append(self.convolve(np.concatenate([data_old, data_new]), h)[n:])
                data_old = data_new[-n:]
            except IndexError:
                time.sleep(0.1)
        # new_data = r.convolve(np.concatenate(scan_plot_data[-len(h):], data), h)[len(h):]
        # scan_plot_data[-shift:, :] = new_data

    def show(self, data_queue: named_deque, figax=None, show=True):
        log.info('Showing audio')

        if figax:
            fig, ax = figax
        else:
            fig, ax = plt.subplots(nrows=1,)

        self.plotdata[data_queue.id] = (np.zeros(AUDIO_PLOT_LENGTH))

        ax.axis((0, len(self.plotdata[data_queue.id]), -1, 1))
        lines = ax.plot(self.plotdata[data_queue.id])

        def update_plot(frame):
            """This is called by matplotlib for each plot update.

            Typically, audio callbacks happen more frequently than plot updates,
            therefore the queue tends to contain multiple blocks of audio data.
            """
            while True:
                try:
                    data = data_queue.popleft()
                except IndexError:
                    break
                shift = len(data)
                self.plotdata[data_queue.id] = np.roll(self.plotdata[data_queue.id], -shift, axis=0)
                self.plotdata[data_queue.id][-shift:] = data

            for column, line in enumerate(lines):
                line.set_ydata(self.plotdata[data_queue.id])
            return lines

        self.ani.append(FuncAnimation(fig, update_plot, interval=30, blit=True))
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

        if N < n:
            raise ValueError("Filter longer than provided data")
        h = np.append(h, np.zeros(N-n))
        h = np.fft.rfft(h)
        data = np.fft.rfft(data)
        conv = h*data
        conv = np.fft.irfft(conv)
        return conv

    # Process queue making sure to have overlap between chunks, identify given signal
    # To identify convolve with reversed signal filter, detect if over threshold, locate peak location

    # If signal is identified save queue from peak for specified length of time to array


class Transmitter(Transceiver):

    def __init__(self):
        Transceiver.__init__(self)
        self.q = multiprocessing.Queue()
        self.blocksize = 1024
        self.transmitting_flag = False
        self.pulser_thread = None

    # Has log colour/tag
    pass

    def play_wav(self, file_name, blocking=False):
        data, fs = sf.read(file_name)
        status = sd.play(data, fs, blocking=blocking)
        if status:
            log.error('Error during playback: ' + str(status))

    def transmit_stream(self):
        def callback(outdata, frames, time, status):
            if status.output_underflow:
                print('Output underflow: increase blocksize?', file=sys.stderr)
                raise sd.CallbackAbort
            assert not status
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                print('Buffer is empty: increase buffersize?', file=sys.stderr)
                raise sd.CallbackAbort
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
                raise sd.CallbackStop
            else:
                print(np.shape(data))
                print(np.shape(outdata))
                outdata[:] = data[:]
        event = threading.Event()
        stream = sd.RawOutputStream(
            samplerate=self.sig.sr, blocksize=self.blocksize,
            channels=1, callback=callback, finished_callback=event.set)
        with stream:
            timeout = 5
            self.q.put(self.sig.get_sinewave(100, 4096), timeout=timeout)
            event.wait()  # Wait until playback is finished

    def play_rand_sync_pulses(self):
        self.transmitting_flag = True
        self.pulser_thread = multiprocessing.Process(target=self.pulser, args=['sync.wav'])
        self.pulser_thread.start()

    def pulser(self, filename):
        while self.transmitting_flag:
            self.play_wav(filename, blocking=True)
            time.sleep(random.randint(0, 5))

    def stop(self):
        log.info("STOPPED TRANSMITTING")
        self.transmitting_flag = False
        self.pulser_thread.join()

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


if __name__ == '__main__':
    log.info('MAIN')
    multiprocessing.freeze_support()

    r, t = initialise()

    sig1 = t.sig.get_sinewave(400, 4000)
    plt.subplot(521)
    plt.plot(sig1)
    fft1 = np.fft.fft(sig1)
    plt.subplot(522)
    plt.plot(fft1)

    sig2 = t.sig.modulate(sig1, 10000, 1)
    plt.subplot(523)
    plt.plot(sig2)
    fft2 = np.fft.fft(sig2)
    plt.subplot(524)
    plt.plot(fft2)

    sig3 = t.sig.modulate(sig2, 10000, 1)
    plt.subplot(525)
    plt.plot(sig3)
    fft2 = np.fft.fft(sig3)
    plt.subplot(526)
    plt.plot(fft2)

    sig4 = t.sig.bias(sig3)
    plt.subplot(527)
    plt.plot(sig4)
    fft3 = np.fft.fft(sig4)
    plt.subplot(528)
    plt.plot(fft3)

    sig5 = t.sig.mean_zero(t.sig.lowpass(sig3, 'egg'))
    plt.subplot(529)
    plt.plot(sig5)
    fft4 = np.fft.fft(sig5)
    plt.subplot(5, 2, 10)
    plt.plot(fft4)


    plt.show()

    # q1 = named_deque()
    # q2 = named_deque()
    # q3 = named_deque()
    #
    #
    # h = r.sig.normalize(r.sig.get_sync_pulse())
    #
    # fig, ax = plt.subplots(nrows=2, sharex='all')
    # ax0 = ax[0]
    # ax1 = ax[1]
    # r.record([q1, q2])
    #
    # r.scan_queue(q2, q3, h)
    # t.play_rand_sync_pulses()
    # t.play_wav('sync.wav')
    #
    # r.show(q3, (fig, ax1), show=False)
    # r.show(q1, (fig, ax0), show=False)
    #
    # plt.show()
    #
    # r.stop()
    # t.stop()
