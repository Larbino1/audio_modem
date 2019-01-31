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
        self.blocksize = 2 ** 12
        self.blockcount = 0
        self.recording_flag = False
        self.recording_steam = sd.InputStream(channels=1, samplerate=self.sig.sr, callback=self.audio_callback, blocksize=self.blocksize)
        self.threads = []

        # PLOTTING
        self.ani = []
        self.plotdata = dict()

        # DEMODULATING
        self.demodulator = demodulator

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
        self.recording_steam.start()
        self.recording_flag = True
        self.threads.append(threading.Thread(target=self.allocator, args=[destinations]))
        self.threads[-1].start()

    def allocator(self, destinations):
        """
        Takes the raw input audio queue and splits it between several destination
        named_deques for various processing tasks
        """
        log.debug('STARTING ALLOCATING')
        destinations.append(self.demodulator.audio_data_queue)
        while self.recording_flag:
            try:
                item = self.q.get()
                for target in destinations:
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

    def scan_queue(self, data_queue, output_filtered_queue, h, threshold=0.25, ):
        log.info("Scanning queue")
        self.convolve_queue(data_queue, output_filtered_queue, h)

        self.threads.append(threading.Thread(target=self.peak_finder, args=(output_filtered_queue, threshold, 2 * len(h))))
        self.threads[-1].start()

    def peak_finder(self, input_queue: deque, threshold, search_width: int):
        N = search_width
        data_old = np.zeros(N)
        while self.recording_flag:
            try:
                data_new, block_num = input_queue.popleft()
                data = np.concatenate((data_old, data_new))
                data_old = data[-N:]
                n = np.argmax(data)
                # If max is in valid range
                if n < (len(data)-N):
                    if data[n] > threshold:
                        log.info('Sync pulse detected')
                        # Start of signal located at n - 2 * len(h1)
                        for i in range(10):
                            if n - search_width + self.blocksize * i >= 0:
                                self.demodulator.demodulate((block_num-i, n - search_width + self.blocksize * i ))
                                break
                        else:
                            log.error('Failed to send peak location')
            except IndexError:
                time.sleep(0.25)

    def action(self, block_num, n):
        log.warning(f"SYNC PULSE DETECTED: blocknum {block_num}, n {n}")

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

    def collapse_queue(self, q):
        log.warning("SHOULD NOT WORK WITH UPDATED QUEUES GIVING BLOCK NUMBERS")
        l = np.array(q)
        l = np.concatenate(l)
        l = np.transpose(l)
        if len(l) == 1:
            l = l[0]
        return l

    # Process queue making sure to have overlap between chunks, identify given signal
    # To identify convolve with reversed signal filter, detect if over threshold, locate peak location

    # If signal is identified save queue from peak for specified length of time to array


class Demodulator:
    def __init__(self):
        self.sig = Signals()
        self.freq = 4000

        self.queue_max_len = 1025
        self.audio_data_queue = named_deque(maxlen=self.queue_max_len)
        self.transmission_start_index = deque()

        self.audio = []
        self.data_bits = []

        # self.running_flag = True
        # self.thread = threading.Thread(target=self.demodulate)
        # self.thread.start()

    # TODO change so only called when a synchronisation pulse is detected, called by child PAM_Demodulator, and returns first bit of data, leaving rest of queue for child

    def find_transmission_start(self):
        """
        Receives a transmission start index and clears the audio queue up to that point,
        returning the first semi-chunk of audio data
        """
        index, n = self.transmission_start_index.popleft()
        log.info(f'Demodulating starting at block {index}, n {n}')

        data = None
        blocknum = 0

        data, blocknum = self.audio_data_queue.popleft()
        # return data, blocknum

        for i in range(self.queue_max_len):
            try:
                data, blocknum = self.audio_data_queue.popleft()
                if blocknum == index:
                    log.debug(f'Returning block {blocknum}')
                    data = data[n:]
                    return data, blocknum
                elif blocknum < index:
                    log.debug(f'Discarding block {blocknum}')
                elif blocknum > index:
                    log.error('Failed to find transmission start in demodulation audio queue')
                    return None
            except IndexError:
                time.sleep(0.5)


class PamDemodulator(Demodulator):
    def __init__(self):
        super().__init__()

    def pam_demod(self, audio, pulse_width, pulse_count):
        ret = np.zeros(pulse_count)
        for k in range(pulse_count):
            if np.sum(audio[k*pulse_width:(k+1)*pulse_width]) > 0:
                ret[k] = 1
        return ret

    def demodulate(self, transmission_start_index,):
        self.transmission_start_index.append(transmission_start_index)
        data, block_index = self.find_transmission_start()

        # Phy level data of fixed length:
        # Get phy level data i.e. symbol_width, symbol count

        while len(data) < 16*2 * 1024:
            try:
                data_new, block_num = self.audio_data_queue.popleft()
                block_index += 1
                log.debug(block_num)
                # assert block_num == block_index, f'LOST AN AUDIO BLOCK, found block {block_num}, expected block {block_index}'
                data = np.append(data, data_new)
            except IndexError:
                time.sleep(0.5)

        # Demodulate phy data
        phy_data = self.sig.amplitude_modulate(data[:1024*32], self.freq, m=0)
        phy_data = self.sig.bias(phy_data)
        phy_data = self.sig.lowpass(phy_data, self.freq//2)
        phy_data = self.sig.mean_zero(phy_data)
        print(self.pam_demod(phy_data, pulse_width=1024, pulse_count=16*2,))

        # Calculate no of audio blocks required

        # Start demodulating thread up to endblock, appending to self.data_bits

        self.test = transmission_start_index
        self.data_bits = phy_data
        log.info('Demodulated')
