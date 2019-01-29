from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


def initialise(modulation=(PAM_Demodulator(), PAM_Modulator())):
    r = Receiver(modulation[0])
    t = Transmitter(modulation[1])
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
    h1 = r.sig.normalize(r.sig.get_sync_pulse())[::-1]

    fig, ax = plt.subplots(nrows=2, sharex='all')
    ax0 = ax[0]
    ax1 = ax[1]

    raw_audio = named_deque()
    scan_data = named_deque()
    filter_data = named_deque()

    r.record([raw_audio, scan_data, filter_data])

    scan_out_queue = named_deque()
    filter_out_queue = named_deque()

    r.scan_queue(scan_data, scan_out_queue, h1, threshold=0.5)
    r.convolve_queue(filter_data, filter_out_queue, h1)

    # t.play_rand_pulses('transmit.wav')
    t.transmit(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]*3))

    r.show(raw_audio, (fig, ax0), show=False, interval=500)
    r.show(filter_out_queue, (fig, ax1), show=False, interval=500)

    plt.show()

    time.sleep(1)

    plt.plot(r.demodulator.data_bits)
    plt.axvline(r.demodulator.test[0] * r.blocksize + r.demodulator.test[1] - 2*len(h1))
    plt.show()

    r.stop()
    t.stop()



