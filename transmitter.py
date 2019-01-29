import sys
import queue
import threading
import multiprocessing
import time
import random

from transceiver import *


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

    # TODO
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
        self.pulser_thread = threading.Thread(target=self.pulser, args=['sync.wav'])
        self.pulser_thread.start()

    def pulser(self, filename):
        while self.transmitting_flag:
            log.debug('Sent pulse')
            self.play_wav(filename)
            time.sleep(random.random() + 0.75)

    def stop(self):
        log.info("STOPPING TRANSMITTING")
        self.transmitting_flag = False
        self.pulser_thread.join(timeout = 5)
        log.info("STOPPED TRANSMITTING")

    # Produce audio clip with leading silence and synchronisation signal

    # Play audio clip on speaker


class Modulator:
    def __init__(self):
        self.sig = Signals()

    @abc.abstractmethod
    def modulate(self, data_bit_array: np.ndarray):
        """
        Receives an array of bits and returns an audio array
        Also adds any modulation specific audio such as calibration symbols etc
        """


class PAM(Modulator):
    def get_pulse(self):
        pass

    def modulate(self, data_bit_array: np.ndarray):
        pass

    # AM TEST
    # sig1 = t.sig.get_sinewave(400, 4000)
    # plt.subplot(521)
    # plt.plot(sig1)
    # fft1 = np.fft.fft(sig1)
    # plt.subplot(522)
    # plt.plot(fft1)
    #
    # sig2 = t.sig.modulate(sig1, 10000, 1)
    # plt.subplot(523)
    # plt.plot(sig2)
    # fft2 = np.fft.fft(sig2)
    # plt.subplot(524)
    # plt.plot(fft2)
    #
    # sig3 = t.sig.modulate(sig2, 10000, 1)
    # plt.subplot(525)
    # plt.plot(sig3)
    # fft2 = np.fft.fft(sig3)
    # plt.subplot(526)
    # plt.plot(fft2)
    #
    # sig4 = t.sig.bias(sig3)
    # plt.subplot(527)
    # plt.plot(sig4)
    # fft3 = np.fft.fft(sig4)
    # plt.subplot(528)
    # plt.plot(fft3)
    #
    # sig5 = t.sig.mean_zero(t.sig.lowpass(sig3, 8000))
    # plt.subplot(529)
    # plt.plot(sig5)
    # fft4 = np.fft.fft(sig5)
    # plt.subplot(5, 2, 10)
    # plt.plot(fft4)
    #
    # plt.show()

