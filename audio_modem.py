from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


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

    q1 = named_deque()
    q2 = named_deque()
    q3 = named_deque()


    h = r.sig.normalize(r.sig.get_sync_pulse())

    fig, ax = plt.subplots(nrows=2, sharex='all')
    ax0 = ax[0]
    ax1 = ax[1]
    r.record([q1, q2])

    r.scan_queue(q2, q3, h)
    t.play_rand_sync_pulses()
    t.play_wav('sync.wav')

    r.show(q3, (fig, ax1), show=False)
    r.show(q1, (fig, ax0), show=False)

    plt.show()

    r.stop()
    t.stop()
