import log
import sys
import queue
import threading
import multiprocessing
import time

import numpy as np
import sounddevice as sd


from transceiver import *

AUDIO_PLOT_LENGTH = 10 * 500 * 44100 // 1000


class Receiver(Transceiver):
    # Has specific log color/tag
    def __init__(self, demodulator):
        Transceiver.__init__(self)

        # RECORDING
        self.q = multiprocessing.Queue()
        self.audio_block_size = self.defaults['audio_block_size']
        self.blockcount = 0
        self.recording_flag = False
        self.recording_steam = sd.InputStream(channels=1, samplerate=self.sig.sr, callback=self.audio_callback, blocksize=self.audio_block_size)
        self.threads = []
        self.allocator_destinations = []

        # PLOTTING
        self.ani = []
        self.plotdata = dict()

        # DEMODULATING
        self.demodulator = demodulator

    #######################################
    # Level 0 - low level audio manip
    #######################################

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        # Fancy indexing with mapping creates a (necessary!) copy:
        if status:
            log.error('audio_callback error:')
            log.error(status)
        self.q.put(indata[:])

    def record(self, *args):
        self.allocator_destinations.extend(args)

        if not self.recording_flag:
            log.info("STARTED RECORDING")
            self.recording_steam.start()
            self.recording_flag = True
            self.threads.append(threading.Thread(target=self.allocator))
            self.threads[-1].start()

    def allocator(self,):
        """
        Takes the raw input audio queue and splits it between several destination
        named_deques for various processing tasks
        """
        log.debug('STARTING ALLOCATING')
        # destinations.append(self.demodulator.audio_data_queue)
        while self.recording_flag:
            try:
                item = self.q.get()
                for target in self.allocator_destinations:
                    assert type(target) == named_deque
                    target.append((np.concatenate(item), self.blockcount))
                self.blockcount += 1
            except queue.Empty:
                time.sleep(0.25)
            if self.q.qsize() > 128:
                log.warning(f'Recording queue backing up, qsize {self.q.qsize()}')
        log.debug('STOPPING ALLOCATING')

    def stop(self):
        log.info("STOPPING RECORDING")
        self.recording_steam.stop()
        self.recording_flag = False
        for t in self.threads:
            t.join(timeout=1)
        self.blockcount = 0
        log.info("STOPPED RECORDING")

    ###############################################
    # Level 1 - mid level functions
    ###############################################

    def scan_queue(self, data_queue, output_queue, h, threshold=0.25, ):
        """
        Filters a queue with h and returns a block number and position if a peak
        of about a certain threshold is found, then quits

        Deletes entries in the queue if no peak is found in the following 10 blocks
        """
        log.info("Scanning queue")
        filtered_signal = named_deque()
        self.convolve_queue(data_queue, filtered_signal, h)

        self.threads.append(threading.Thread(target=self.peak_finder, args=(filtered_signal, threshold, 2 * len(h), output_queue)))
        self.threads[-1].start()

    def peak_finder(self, input_queue: deque, threshold, search_width: int, output_queue=None):
        N = search_width
        data_old = np.zeros(N)
        # buffer = named_deque(maxlen=10)
        while self.recording_flag:
            try:
                # Get next chunk from queue
                data_new, block_num = input_queue.popleft()
            except IndexError:
                time.sleep(0.25)
                continue

            # Extend with previous data for convolution
            data = np.concatenate((data_old, data_new))
            data_old = data[-N:]

            # Find the index of the peak
            n = np.argmax(data)

            # Make sure that the peak is in the valid convolution region
            if n < (len(data)-N) and data[n] > threshold:
                log.info('Sync pulse detected')
            else:
                # If peak in in right hand invalid region, get next chunk of data
                continue

            # Get block number and sample index of star of signal (at n-search_width)
            for i in range(10):
                if n - search_width + self.audio_block_size * i >= 0:
                    transmission_start = (block_num - i, n - search_width + self.audio_block_size * i)
                    log.special(f'Peak detected at {transmission_start}')
                    break
            else:
                log.error('Failed to send peak location')
                continue

            if type(output_queue) == named_deque:
                output_queue.append(transmission_start)

    def convolve_queue(self, data_queue, output_queue, h,):
        self.threads.append(threading.Thread(target=self.convolver, args=(data_queue, output_queue, h)))
        self.threads[-1].start()

    def convolver(self, data_queue, output_queue, h):
        n = len(h)
        data_old = np.zeros(n)
        while self.recording_flag:
            try:
                data_new, block_num = data_queue.popleft()
                conv = self.sig.convolve(np.concatenate([data_old, data_new]), h)[n:]
                output_queue.append((conv, block_num))
                data_old = data_new[-n:]
            except IndexError:
                time.sleep(0.25)

    def demodulate_queue(self, audio_queue, transmission_index_queue, output_queue):
        self.threads.append(threading.Thread(
                target=self.demodulator_thread_function, args=(audio_queue, transmission_index_queue, output_queue)))
        self.threads[-1].start()

    def demodulator_thread_function(self, audio_queue, transmission_index_queue, output_queue):
        while self.recording_flag:
            try:
                block_num, n = transmission_index_queue.popleft()
                self.demodulator.demodulate((block_num, n), audio_queue, output_queue)
            except IndexError:
                time.sleep(0.1)
                continue

    ########################################
    # Level 2 - user end functions
    ########################################

    def listen(self, output_queue, threshold=1.0):
        """
        Listens for the sync pulse and when detected, demodulates and decodes it, outputting it to a queue of
        data bits
        """
        if not self.recording_flag:
            log.warning("Listening but not recording, ensure receiver.record() has been called")
        demod_audio_queue = named_deque(64)
        scan_audio_queue = named_deque(64)
        self.allocator_destinations.extend([
            demod_audio_queue,
            scan_audio_queue
        ]
        )
        transmission_indices_queue = named_deque()
        demodulated_chunks = named_deque()

        q = self.sig.get_sync_pulse_matched_filter()
        self.scan_queue(scan_audio_queue, transmission_indices_queue, q, threshold=threshold)
        self.demodulate_queue(demod_audio_queue, transmission_indices_queue, output_queue)

    def listen_for_text(self, threshold=0.5):
        self.record()
        output_queue = named_deque()
        self.listen(output_queue, threshold)
        while True:
            try:
                received_bits = np.concatenate(list(output_queue))
                output_queue.clear()
                received_bytes = np.packbits(received_bits)
                text = bytes(received_bytes)
                log.special(text)
            except:
                time.sleep(5)

    ########################################
    # Level 3 - analysis
    ########################################

    def show(self, data_queue: named_deque, figax=None, show=True, interval=30):
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
                    data, block_num = data_queue.popleft()
                except IndexError:
                    break
                shift = len(data)
                self.plotdata[data_queue.id] = np.roll(self.plotdata[data_queue.id], -shift, axis=0)
                self.plotdata[data_queue.id][-shift:] = data

            for column, line in enumerate(lines):
                line.set_ydata(self.plotdata[data_queue.id])
            return lines

        self.ani.append(FuncAnimation(fig, update_plot, interval=interval, blit=True))
        if show:
            plt.show()


class Demodulator(Transceiver):
    def __init__(self):
        super().__init__()
        self.freq = 4000
        self.queue_max_len = 1024

        self.audio = []
        self.data_bits = []

        # List used for passing debug data to main thread etc.
        self.demodulated_flag = False
        self.debug_data = []
        self.test = []

    def find_transmission_start(self, transmission_start_index, audio_data_queue):
        """
        Receives a transmission start index and clears the audio queue up to that point,
        returning the first semi-chunk of audio data
        """
        index, n = transmission_start_index
        log.info(f'Demodulating starting at block {index}, n {n}')

        for i in range(self.queue_max_len):
            try:
                data, blocknum = audio_data_queue.popleft()
                if blocknum == index:
                    # log.debug(f'Returning block {blocknum}')
                    data = data[n:]
                    return data, blocknum
                elif blocknum < index:
                    pass
                    # log.debug(f'Discarding block {blocknum}')
                elif blocknum > index:
                    log.error('Failed to find transmission start in demodulation audio queue')
                    return None
            except IndexError:
                time.sleep(0.5)


class PamDemodulator(Demodulator):
    def __init__(self):
        super().__init__()

    def pam_demod(self, audio, pulse_width, pulse_count):
        ret = np.zeros(pulse_count, dtype=np.uint8)
        for k in range(pulse_count):
            if np.sum(audio[k*pulse_width:(k+1)*pulse_width]) > 0:
                ret[k] = 1
        return ret

    def demodulate(self, transmission_start_index, audio_data_queue, output_queue):
        pass


class AmPamDemodulator(PamDemodulator):
    def __init__(self):
        super().__init__()

    def ampam_demod(self, data, pulse_width, pulse_count=None, mean=None, debug=False):
        """
        Decodes up to pulse count bits. If no pulse count is provided, decodes up to
        the length of the data.
        Returns a list of bits, and any trailing data not decoded
        """

        N = len(data)
        assert N > pulse_width, 'Not enough audio data, must be at least one pulse width'

        pc = pulse_count
        pw = pulse_width
        if not pc:
            pc = N//pw
        # Demodulate phy data
        end_data = data[pc*pw:]
        data = self.sig.amplitude_modulate(data[:pc*pw], self.freq, m=0)
        data = self.sig.bias(data)
        data = self.sig.lowpass(data, self.freq//2)
        if not mean:
            mean = np.mean(data)
        data = data - mean
        bits = self.pam_demod(data, pulse_width=pw, pulse_count=pc,)
        if debug:
            self.debug_data.append((data, bits))
        return end_data, bits

    def demodulate(self, transmission_start_index, audio_data_queue, output_queue):
        log.special(transmission_start_index)
        data, start_block_index = self.find_transmission_start(transmission_start_index, audio_data_queue)

        # Phy level data of fixed length:
        # Get phy level data i.e. symbol_width, symbol count

        audio_block_size        = self.defaults['audio_block_size']
        initial_pulse_width     = self.defaults['ampam']['initial_pulse_width']
        threshold_data_bits     = self.defaults['ampam']['threshold_data_bits']
        pulse_width_data_bits   = self.defaults['ampam']['pulse_width_data_bits']
        pulse_count_data_bits   = self.defaults['ampam']['pulse_count_data_bits']
        initial_pulse_count     = pulse_width_data_bits + pulse_count_data_bits + threshold_data_bits

        # TODO make into a function
        i = start_block_index
        while len(data) < initial_pulse_count * initial_pulse_width:
            try:
                data_new, block_num = audio_data_queue.popleft()
                i += 1
                assert block_num == i, f'LOST AN AUDIO BLOCK, found block {block_num}, expected block {start_block_index}'
                data = np.append(data, data_new)
            except IndexError:
                time.sleep(0.5)

        data, phy_bits = self.ampam_demod(data, initial_pulse_width, initial_pulse_count, mean=None)

        # Calculate no of audio blocks required
        # TODO manage this better
        pulse_width_bytes = np.packbits(phy_bits[threshold_data_bits:threshold_data_bits+pulse_width_data_bits])
        pulse_count_bytes = np.packbits(phy_bits[threshold_data_bits+pulse_width_data_bits:threshold_data_bits+pulse_width_data_bits+pulse_count_data_bits])

        pulse_width = pulse_width_bytes[0]*2**8 + pulse_width_bytes[1]
        pulse_count = pulse_count_bytes[0]*2**8 + pulse_count_bytes[1]
        log.debug(f'Receiving pam with pulse_count {pulse_count} pulse_width {pulse_width}')

        bits_decoded = 0
        total_transmission_length = initial_pulse_count * initial_pulse_width + pulse_count * pulse_width
        end_block_index = np.ceil(start_block_index + total_transmission_length / audio_block_size)
        log.debug(f'Calculated end block index = {end_block_index}, current block_num {block_num}')
        while block_num < end_block_index+1:
            try:
                data_new, block_num = audio_data_queue.popleft()
                i += 1
                assert block_num == i, f'LOST AN AUDIO BLOCK, found block {block_num}, expected block {start_block_index}'
                data = np.append(data, data_new)
            except IndexError:
                if len(data) > 100 * pulse_width:
                    # TODO bayesian updating of mean, incorporating new data, and old data prior to give posterior point estimate for mean
                    data, bits = self.ampam_demod(data, pulse_width, mean=None)
                    if bits_decoded + len(bits) > pulse_count:
                        output_queue.append(bits[:pulse_count-bits_decoded])
                        break
                    bits_decoded += len(bits)
                    output_queue.append(bits)
                else:
                    time.sleep(0.1)
        self.demodulated_flag = True
        log.info('Demodulated')
