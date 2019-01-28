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


def play_and_record_wav(wav_filename, save_filename):
    q1 = deque()
    r, t = initialise()
    r.record([q1])
    t.play_wav(wav_filename)
    time.sleep(sf.info(wav_filename).duration + 0.5)
    r.stop()
    data = r.collapse_queue(q1)
    plt.plot(data)
    plt.show()
    for i in range(5):
        try:
            start = int(input('Enter start index: >>'))
            end = int(input('Enter end index: >>'))
            break
        except Exception as e:
            print(f'ERROR PARSING INPUT TRY AGAIN: {e}')
    plt.plot(data[start:end])
    plt.show()
    r.sig.save_csv_data(save_filename, data[start:end])


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

    q = r.sig.get_channel_response()
    q_reversed = q[::-1]
    h1 = r.sig.normalize(r.sig.get_sync_pulse())[::-1]
    h2 = r.sig.normalize(r.sig.convolve(h1, q))
    h3 = r.sig.normalize(r.sig.convolve(h1, q[::-1]))

    plt.subplot(121)
    plt.plot(h1)
    plt.subplot(122)
    plt.plot(h2)

    fig, ax = plt.subplots(nrows=4, sharex='all')
    ax0 = ax[0]
    ax1 = ax[1]
    ax2 = ax[2]
    ax3 = ax[3]

    raw_audio = named_deque()
    filter_data_1 = named_deque()
    filter_data_2 = named_deque()
    filter_data_3 = named_deque()

    r.record([raw_audio, filter_data_1, filter_data_2, filter_data_3])

    filtered_data_1 = named_deque()
    filtered_data_2 = named_deque()
    filtered_data_3 = named_deque()

    r.scan_queue(filter_data_1, filtered_data_1, h1)
    r.scan_queue(filter_data_2, filtered_data_2, h2)
    r.scan_queue(filter_data_3, filtered_data_3, h3)

    r.show(raw_audio, (fig, ax0), show=False, interval=500)
    r.show(filtered_data_1, (fig, ax1), show=False, interval=500)
    r.show(filtered_data_2, (fig, ax2), show=False, interval=500)
    r.show(filtered_data_3, (fig, ax3), show=False, interval=500)

    # for ax in [ax0, ax1, ax2]:
    #     ax.axis((0, 44100, -0.25, 0.25))

    t.play_rand_sync_pulses()

    time.sleep(2)

    r.stop()
    t.stop()

    plt.show()

