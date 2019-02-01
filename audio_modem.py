from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


def initialise(modulation=(AmPamDemodulator(), AmPamModulator())):
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

random.seed(100)
# TODO test with seeded random packet
# test_packet = Packet([1, 0, 1, 0, 1, 0, 0, 0, 0]*3)
test_packet = Packet([(random.getrandbits(1)) for i in range(100)])

if __name__ == '__main__':
    log.info('MAIN')
    multiprocessing.freeze_support()

    r, t = initialise()

    fig, ax = plt.subplots(nrows=2, sharex='all')
    ax0 = ax[0]
    ax1 = ax[1]

    raw_audio = named_deque()
    listen_queue = named_deque()
    filter_data = named_deque()

    r.record(raw_audio, listen_queue, filter_data)

    filter_output_queue = named_deque()

    r.listen(threshold=0.5)
    r.convolve_queue(filter_data, filter_output_queue, r.sig.get_sync_pulse_matched_filter())

    # t.play_rand_pulses('sync.wav')
    t.transmit(test_packet)

    r.show(raw_audio, (fig, ax0), show=False, interval=500)
    r.show(filter_output_queue, (fig, ax1), show=False, interval=500)

    plt.show()

    plt.plot()
    plt.plot(r.demodulator.data_bits)
    plt.show()

    r.stop()
    t.stop()



