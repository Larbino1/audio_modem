import sys
import queue
import threading
import multiprocessing
import time
import random

from transceiver import *


class Modulator:
    def __init__(self):
        self.sig = Signals()
        self.audio = []

    @abc.abstractmethod
    def modulate(self,  _data_bit_array: np.ndarray):
        """
        Receives an array of bits and returns an audio array
        Also adds any modulation specific audio such as calibration symbols etc
        """
        self.audio.append(np.zeros(int(0.1*self.sig.sr)))
        self.audio.append(self.sig.get_sync_pulse())


class PAM_Modulator(Modulator):
    def get_pulse(self):
        pass

    def modulate(self, data_bit_array: np.ndarray):
        super(PAM_Modulator, self).modulate(data_bit_array)

        N = 2**10
        baseband_signal = []
        for bit in data_bit_array:
            if bit == 1:
                baseband_signal.extend(np.ones(N))
            else:
                baseband_signal.extend(np.zeros(N))
        am_signal = baseband_signal * self.sig.get_sinewave(5000, len(baseband_signal))
        self.audio.append(am_signal)

        return np.concatenate(self.audio)


class Transmitter(Transceiver):

    def __init__(self, modulator: Modulator):
        Transceiver.__init__(self)

        self.modulator = modulator

        self.q = multiprocessing.Queue()
        self.blocksize = 1024
        self.transmitting_flag = False
        self.threads = []

    # Has log colour/tag
    pass

    def transmit(self, data_bit_array):
        audio = self.modulator.modulate(data_bit_array)
        self.sig.save_array_as_wav('transmit.wav', audio)
        self.play_wav('transmit.wav')

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

    def play_rand_pulses(self, filename):
        self.transmitting_flag = True
        self.threads.append(threading.Thread(target=self.pulser, args=[filename]))
        self.threads[-1].start()

    def pulser(self, filename):
        while self.transmitting_flag:
            log.debug('Sent pulse')
            self.play_wav(filename)
            time.sleep(random.random() + 3)

    def stop(self):
        log.info("STOPPING TRANSMITTING")
        self.transmitting_flag = False
        for t in self.threads:
            t.join(timeout = 5)
        log.info("STOPPED TRANSMITTING")

    # Produce audio clip with leading silence and synchronisation signal

    # Play audio clip on speaker

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

