from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


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

    q = r.sig.get_channel_response()
    q_reversed = q[::-1]
    h = r.sig.normalize(r.sig.get_sync_pulse())[::-1]
    h1 = r.sig.normalize(r.sig.convolve(h, q[::-1]))

    fig, ax = plt.subplots(nrows=2, sharex='all')
    ax0 = ax[0]
    ax1 = ax[1]

    raw_audio = named_deque()
    filter_data_1 = named_deque()

    r.record([raw_audio, filter_data_1,])

    filtered_data_1 = named_deque()

    r.scan_queue(filter_data_1, filtered_data_1, h1, threshold=0.05, plotting=False)

    r.show(raw_audio, (fig, ax0), show=False, interval=500)
    # r.show(filtered_data_1, (fig, ax1), show=False, interval=500)

    t.play_rand_sync_pulses()

    plt.show()

    r.stop()
    t.stop()


