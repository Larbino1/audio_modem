import log
import sys
import queue
import threading
import multiprocessing
import time

import numpy as np
import sounddevice as sd


from transceiver import *

AUDIO_PLOT_LENGTH = 3 * 500 * 44100 // 1000


class Receiver(Transceiver):
    # Has specific log color/tag
    def __init__(self):
        Transceiver.__init__(self)
        self.q = multiprocessing.Queue()
        self.blocksize = 2 ** 12
        self.blockcount = 0
        self.recording_flag = False
        self.stream = sd.InputStream(channels=1, samplerate=self.sig.sr, callback=self.audio_callback, blocksize=self.blocksize)
        self.alloc_thread = None
        self.threads = []
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
                    target.append((np.concatenate(item), self.blockcount))
                    self.blockcount += 1
            except queue.Empty:
                time.sleep(0.25)
            if self.q.qsize() > 128:
                log.warning(f'Recording queue backing up, qsize {self.q.qsize()}')
        log.debug('STOPPING ALLOCATING')

    def stop(self):
        log.info("STOPPING RECORDING")
        self.stream.stop()
        self.recording_flag = False
        self.alloc_thread.join(timeout=1)
        for t in self.threads:
            t.join(timeout=1)
        self.blockcount = 0
        log.info("STOPPED RECORDING")

    def scan_queue(self, data_queue, output_queue, h, threshold=0.25, plotting=False):
        self.threads.append(threading.Thread(
            target=self.convolver,
            args=(data_queue, output_queue, h)))
        self.threads[-1].start()
        if not plotting:
            self.threads.append(threading.Thread(
                target=self.peak_finder,
                args=(output_queue, threshold, 2 * len(h))))
            self.threads[-1].start()

    def peak_finder(self, input_queue: deque, threshold, search_width: int):
        N = search_width
        data_old = np.zeros(N)
        while self.recording_flag:
            try:
                data_new, block_num= input_queue.popleft()
                data = np.concatenate((data_old, data_new))
                n = np.argmax(data)
                # If max is in valid range
                if n < len(data)-N:
                    if data[n] > threshold:
                        self.action(block_num, n)
                data_old = data[-N:]
            except IndexError:
                time.sleep(0.25)

    def action(self, block_num, n):
        log.warning(f"SYNC PULSE DETECTED: blocknum {block_num}, n {n}")

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
        log.info('Stopping showing audio')

    def collapse_queue(self, q):
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
        pass

    @abc.abstractmethod
    def demodulate(self, audio_array):
        """
        Recevies an audio array and returns an array of bits
        """
