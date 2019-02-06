import sys
import queue
import threading
import multiprocessing
import time
import random

from transceiver import *


class Modulator(Transceiver):
    def __init__(self, channel):
        super().__init__()
        self.audio = []
        self.sync_pulse_index = None

        self.freq = self.channels[channel]['freq']

        self.debug_data =[]

    @abc.abstractmethod
    def modulate(self,  _data_packet: Packet):
        """
        Receives an array of bits and returns an audio array
        Also adds any modulation specific audio such as calibration symbols etc
        """

    # @abc.abstractmethod
    # def get_sync_pulse(self):
    #     log.warning('No sync pulse defined')
    #     return np.ndarray([])


class PamModulator(Modulator):
    def __init__(self, channel):
        super().__init__(channel)
        self.signal = []

    def pam_mod(self, data_bits, pulse_width):
        for bit in data_bits:
            if int(bit) == 1:
                self.audio.append(np.ones(pulse_width))
            else:
                self.audio.append(np.zeros(pulse_width))

    def modulate(self, data_packet: Packet):
        super(PamModulator, self).modulate(data_packet)

        data_bit_array = data_packet.unpack()

        pulse_width = 80
        pulse_count = len(data_bit_array)

        pulse_width_bits = self.bop.binary_repr(pulse_width, width=self.defaults['ampam']['pulse_width_data_bits'])
        pulse_count_bits = self.bop.binary_repr(pulse_count, width=self.defaults['ampam']['pulse_count_data_bits'])

        initial_pulse_width = self.defaults['ampam']['initial_pulse_width']
        # Do not change following line without modifying defaults['ampam']['threshold_data_bits']
        self.pam_mod([0, 1, 0, 1], initial_pulse_width)
        self.pam_mod(pulse_width_bits, initial_pulse_width)
        self.pam_mod(pulse_count_bits, initial_pulse_width)

        log.debug(f'Transmitting pam with pulse_count = {pulse_count}, pulse_width = {pulse_width}')

        self.pam_mod(data_bit_array, pulse_width)
        data = np.concatenate(self.audio)
        if self.debug_mode:
            self.debug_data.append( (data, data_bit_array, pulse_width, pulse_count) )
        return data


class AmPamModulator(PamModulator):
    def __init__(self, channel):
        super().__init__(channel)

    def modulate(self, data_packet: Packet):
        pam_signal = super().modulate(data_packet)
        return self.sig.amplitude_modulate(pam_signal, self.freq)

    # def get_sync_pulse(self):
    #     return self.sig.get_sync_pulse()


class Transmitter(Transceiver):

    def __init__(self, modulator: Modulator, debug_mode=False):
        Transceiver.__init__(self)

        self.modulator = modulator

        self.q = multiprocessing.Queue()
        self.blocksize = 1024
        self.transmitting_flag = False
        self.threads = []

    # Has log colour/tag
    pass

    def transmit(self, packet: Packet):
        audio = self.modulator.modulate(packet)
        audio = np.concatenate([np.zeros(int(0.1 * self.sig.sr)), self.sig.get_sync_pulse(), audio])
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
                log.error('Output underflow: increase blocksize?')
                raise sd.CallbackAbort
            assert not status
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                log.error('Buffer is empty: increase buffersize?')
                raise sd.CallbackAbort
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
                raise sd.CallbackStop
            else:
                outdata[:] = data[:]
        event = threading.Event()
        stream = sd.RawOutputStream(
            samplerate=self.sig.sr, blocksize=self.blocksize,
            channels=1, callback=callback, finished_callback=event.set)
        with stream:
            timeout = 5
            self.q.put(self.sig.get_sinewave(100, 4096), timeout=timeout)
            event.wait()  # Wait until playback is finished

    def stop(self):
        log.info("STOPPING TRANSMITTING")
        self.transmitting_flag = False
        for t in self.threads:
            t.join(timeout = 5)
        log.info("STOPPED TRANSMITTING")


    def play_rand_pulses(self, filename):
        self.transmitting_flag = True
        self.threads.append(threading.Thread(target=self.pulser, args=[filename]))
        self.threads[-1].start()

    def pulser(self, filename):
        while self.transmitting_flag:
            log.debug('Sent pulse')
            self.play_wav(filename)
            time.sleep(random.random() + 3)

    ###
    #
    ###

    def transmit_text(self, text):
        packet = Packet(self.bop.text_to_bits(text))
        self.transmit(packet)

